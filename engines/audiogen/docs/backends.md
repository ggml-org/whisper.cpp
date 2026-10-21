# audiogen engine: backends

Part of the [audiogen engine documentation](../README.md).

## Backends

`n_gpu_layers > 0` (`--gpu` on the CLI) selects a GPU backend through the ggml
registry. Selection tries Adreno 700+ OpenCL first; validated
Vulkan/Metal/CUDA discrete devices, then validated integrated devices; and
finally other discrete, then integrated GPU backends. Integrated-device
support is required because Vulkan reports Android UMA adapters such as
Pixel's Mali-G715 as `IGPU`.

`EngineOptions::backend` (`--backend` on the CLIs) replaces that walk with an
explicit device: `cpu`, `opencl`, `webgpu`, `hexagon` (the Snapdragon HTP0
NPU), or an exact ggml device name. An explicit request has no fallback, so `create()`
throws when the device is missing. The Hexagon NPU is never picked by the
automatic walk; see [Hexagon NPU](#hexagon-npu).
`EngineOptions::lm_backend` (`--lm-backend`) moves only the LM to another
device with the same names.

| Stage | Placement when a GPU is selected |
|---|---|
| DiT, VAE | GPU (or the Hexagon NPU on request) |
| Text encoder, condition encoder | GPU, unless `ACESTEP_ENCODERS_CPU` is present |
| LM | GPU on Vulkan (every device except Mali), Metal, OpenCL and CUDA, the NPU on Hexagon; CPU on Mali Vulkan devices and every unmeasured backend (WebGPU included) |
| FSQ detokenizer | GPU on Vulkan, Metal, OpenCL, and CUDA, the NPU on Hexagon; CPU on every unmeasured backend (WebGPU included) |

The LM and detokenizer are allowlisted per backend: a backend nobody has run keeps the CPU placement, so adding one cannot silently regress generated audio. Metal and OpenCL are validated for both stages; the recorded OpenCL validation used an Adreno 740. Vulkan carries both stages on every device except the ones on a per-device denylist (`vulkan_device_lm_blocked`): Mali keeps the LM on the CPU, because Mali-G715 testing showed code collapse and early termination there. A single misbehaving GPU family should cost that family the stage, not every Vulkan device, so the per-device rule denies rather than admits. CUDA needs the `snake` / `col2im_1d` VAE kernels from the `ggml-speech` fork's CUDA backend and its `GGML_PREC_F32` MUL_MAT support (the LM-shaped q4_0/q4_K strided-B `b_absmax=1e5` stress cases produced NaN before the fork routed explicit-precision matmuls to the f32 cuBLAS path); with that ggml, `test-backend-ops` on an RTX 5090 passes the full suite, the Q8_0 LM matches the F32-dequantized reference at 0.9999 logit cosine with an identical greedy trajectory where the CPU Q8_0 path sits at 0.994, and `--quantized-batch-cfg-regression` passes, so the LM runs on the GPU. Hexagon runs the LM on the NPU, which needs a `ggml-speech` whose HTP `q8_0` activation quantizer matches the CPU's (see [Hexagon NPU](#hexagon-npu)). The HIP/MUSA builds of the same backend register as `ROCm`/`MUSA` and stay off the allowlist until measured.

Measurement is against an F32-dequantized reference (`scripts/dequant_gguf.py`), not against CPU. CPU is not automatically ground truth for a quantized model — ggml's CPU matmul quantizes activations to Q8_1 internally. On Metal the LM reproduces the F32 argmax trajectory exactly where CPU Q8_0 diverges at the first token. The RADV validation (AMD Strix Halo, Radeon 8060S) followed the same protocol: on 300-token prefills the Vulkan Q8_0 LM matches the F32 reference argmax on 3/3 probes with logit cosine >= 0.99999995, where CPU Q8_0 matches on 1/3 at cosine ~0.9998, and the GPU LM stage runs ~2x faster than the CPU path on that device. The NVIDIA Vulkan validation (GeForce RTX 3080, proprietary driver) used the same protocol on 300-token prefills: the Vulkan Q8_0 LM matches the F32 reference argmax on 3/3 seeds with logit cosine >= 0.99999998 and top-50 overlap 49.8/50, where CPU Q8_0 on the same box matches on 0/9, 5/9 and 0/9 steps at cosine ~0.9999 with logit differences up to 10.2. On Mali Vulkan, however, the LM produces repeated semantic codes and can terminate at roughly half the requested duration, while the same device produces a diverse full-length sequence with the LM on CPU — that report predates the strided-`src0` matmul binding fix in the `ggml-speech` Vulkan backend, which removed exactly that degenerate-trajectory symptom on another non-cooperative-matrix GPU, so the Mali entry is worth re-testing on device before it is trusted further.

On a GPU that supports F32 flash attention, Phase 2 also decodes the conditional and unconditional CFG paths in one batched graph. This matches the reference `acestep.cpp` LM path: the same prompt, model, sampler settings, and seed produce the same semantic-code sequence. Unsupported backends keep the separate F32 manual-attention path.

`lm-smoke --gpu --quantized-batch-cfg-regression --model <Q4-or-Q8-LM.gguf>` (or `--backend hexagon` in place of `--gpu`) compares the compact batched head against two full-vocabulary decode streams. It requires quantized tied embeddings and fails if either stream changes argmax or drops below `0.99999` logit cosine.

The policy itself lives in [`src/acestep/stage_placement.h`](../src/acestep/stage_placement.h), separate from the engine, so it is unit tested without a GPU.

### Environment overrides

Applied after the allowlist. The LM, detokenizer, and encoder overrides use
presence semantics: even a value of `0` activates them, and CPU wins when both
CPU and GPU are present for one stage. `ACESTEP_VAE_GPU` checks its first
character: `1` selects GPU and every other present value selects CPU.

| Variable | Effect |
|---|---|
| `ACESTEP_LM_GPU` / `ACESTEP_LM_CPU` | presence forces the LM onto the GPU or CPU |
| `ACESTEP_DETOK_GPU` / `ACESTEP_DETOK_CPU` | presence forces the detokenizer onto the GPU or CPU |
| `ACESTEP_ENCODERS_CPU` | presence moves the encoders to the CPU to trim wired memory |
| `ACESTEP_VAE_GPU` | a value beginning with `1` forces GPU; any other present value forces CPU |
| `ACESTEP_KEEP_STAGES` | values beginning with `1`, `t`, `T`, `y`, or `Y` eagerly load and keep every stage resident |
| `ACESTEP_LM_DUMP_LAYERS` | write the LM's per-layer prefill hidden states to this file path |
| `ACESTEP_LM_DUMP_TOKENS` | write the Phase-2 prompt token IDs as CSV |
| `ACESTEP_LM_DUMP_LOGITS` | write the conditional Phase-2 prefill logits as raw F32 |
| `ACESTEP_VAE_PROFILE` | print the VAE per-op-type time inventory |
| `ACESTEP_VAE_WIN_CORE` | diagnostic-only positive integer that pins the decode window core; not a supported tuning API |

Use `ACESTEP_LM_GPU` or `ACESTEP_DETOK_GPU` to take the measurement that would widen the allowlist for a new backend, without a rebuild.

### Parity debug hooks

`ACESTEP_PARITY_DEBUG` is a compile-time macro, not an environment variable, and there is no CMake option for it: configure with `-DCMAKE_CXX_FLAGS=-DACESTEP_PARITY_DEBUG`. The hooks below are `#ifdef`-ed out of `generate()` without it, so setting them on a default build does nothing.

| Variable | Effect |
|---|---|
| `ACESTEP_DUMP_DIR` | log the tokenizer shapes and write `our_dit_output.bin`, `our_noise.bin`, `our_context.bin`, `our_enc_hidden.bin` here for tensor-level comparison against acestep.cpp |
| `ACESTEP_INJECT_NOISE`, `ACESTEP_INJECT_ENC`, `ACESTEP_INJECT_CONTEXT` | replace the noise, encoder hidden states, or DiT context with an acestep.cpp `--dump` tensor to isolate a diverging stage |

Dumps use a flat `[ndim, d0, d1]` int32 header followed by the `f32` payload, the same format the injection hooks read.

### Memory

By default no stage stays resident after `create()`: `generate()` loads each stage immediately before its step and frees it right after, so only one stage is resident for the LM, detokenizer, DiT, and VAE steps rather than all six at once, which is what keeps a non-entitled iOS app inside its memory budget. The one overlap is the condition encoder: it supplies the silence frame that pads the DiT context before the text encoder runs and its own forward comes after text encoding, so it is loaded first and the two encoders are co-resident until the text encoder is freed. That pair is the only two-stage overlap, not necessarily the largest memory footprint. A truthy `ACESTEP_KEEP_STAGES` opts out, eagerly loads every stage in `Engine::create()`, and keeps them resident. With stages kept resident, the LM and DiT also retain their reusable forward graphs and activation pools between generations (they are freed with the model in the default per-stage mode), trading additional steady-state memory for repeat-generation speed.

Long VAE decodes are split into overlapping windows. The engine probes the
active backend's maximum allocation size against the real decode graph and
shrinks the core window when needed; short inputs remain a single graph.
`ACESTEP_VAE_WIN_CORE` can pin the core only for diagnostics. VAE encode is not
windowed and still allocates one full graph.

## Hexagon NPU

`backend = "hexagon"` runs every stage on the HTP0 session of the
ggml-speech Hexagon backend: the DiT, the VAE, both encoders, the FSQ
detokenizer and the LM. `lm_backend = "cpu"` or `"opencl"` moves only the LM
off the NPU. Requirements:

- ggml-speech built with `GGML_HEXAGON=ON`, with the `libggml-htp-v*.so` DSP
  skeletons staged next to the host backends. `backends_dir` is prepended to
  FastRPC's `DSP_LIBRARY_PATH` before the HTP session opens.
- A ggml-speech whose HTP `q8_0` activation quantizer matches the CPU's
  (an F32 scale per 32 activations). Older skeletons scale 128 activations at
  a time in F16: the LM's prompt logits then land 6x further from the F32
  reference than CPU Q8_0, its greedy decode leaves the reference at the first
  audio code, and activations past 65504 turn into NaN.
- The `q8_0` DiT (`ditVariant: 'turbo-q8'`). The HTP matmuls cover
  `q4_0`/`q8_0`/`mxfp4` and float weights, not the K-quants of the default
  `turbo-q4` (`Q4_K_M`) DiT.

Stage weights are uploaded into the session's repack buffer type instead of
being mapped from the GGUF (HTP computes only on its own buffers), so a stage
load includes a one-time repack. The LM's tied embedding is both a
`GET_ROWS` table and a 217,204-row head; HTP cannot gather rows from its
repacked weights and refuses a `MUL_MAT` past 32,768 rows. On such a backend
the LM looks the embeddings up on the host from the mapped GGUF and projects
the head in row chunks, cut where the Phase-1 prefix head and the Phase-2
compact head begin and end, so neither phase copies the table.

Parity on a Galaxy S25 (Snapdragon 8 Elite, Hexagon v79), cosine against the
CPU backend with each stage fed identical inputs, OpenCL on the same phone
for reference:

| Stage | Hexagon | OpenCL (Adreno 830) |
|---|--:|--:|
| FSQ detokenizer | 0.99999 | 0.99991 |
| text encoder | 0.99941 | 0.99913 |
| condition encoder | 0.99947 | 0.99924 |
| DiT, 8 turbo steps | 0.989 | 0.958 |
| VAE decode, 7.2 s | 0.999992 | 0.999992 |

The DiT row compounds eight flow-matching steps, so it is where every
backend drifts furthest from the CPU; complete renders therefore differ from
the CPU waveform on OpenCL and Hexagon alike, and the per-stage rows isolate
each backend's own error.

The `q8_0` LM on HTP0 is as close to the F32-dequantized reference as CPU
`q8_0`. Over a 69-token Phase-2 prompt and 48 greedy steps it keeps the
reference's greedy trajectory, as the CPU does, with a mean audio-code logit
KL of 1.05e-2 (CPU 1.10e-2, OpenCL 2.6e-6). `--quantized-batch-cfg-regression`
passes with identical logits.

Stage compute on the same phone (`music-cli --dur 30`, `ACESTEP_KEEP_STAGES=1`
so loads are excluded, `q8_0` DiT, the run with the median total of three,
milliseconds):

| Placement | LM | DiT | VAE | generate |
|---|--:|--:|--:|--:|
| CPU (8 threads) | 26,459 | 51,861 | 58,725 | 147,482 |
| OpenCL (Adreno 830) | 10,661 | 10,989 | 28,519 | 51,574 |
| `hexagon` + `lm_backend = "cpu"` | 23,812 | 1,946 | 12,165 | 38,517 |
| `hexagon` + `lm_backend = "opencl"` | 10,830 | 1,870 | 13,440 | 26,686 |
| `hexagon` | 7,522 | 1,920 | 12,458 | 22,479 |

At `--dur 8` the same five placements generate in 45,743 / 13,412 / 16,267 /
8,973 / 7,832 ms.

## MiniMax Core ML conditioner

On Apple Silicon with macOS 13 or later, build with `AUDIOGEN_COREML=ON`
and export the projection from the same F16 synth GGUF used by the engine:

```bash
python engines/audiogen/scripts/export-minimax-conditioner-coreml.py \
  --synth models/minimax/mm3-synth-f16.gguf \
  --out models/minimax/conditioner.mlpackage --frames 150 \
  --compile-dir models/minimax
```

Export requires PyTorch, NumPy, `gguf` and `coremltools`; inference requires
only the native library and Apple frameworks. Request the compiled model in
the C++ API:

```cpp
tts_cpp::minimax::EngineOptions options;
options.model_dir = "models/minimax";
options.device = "gpu";
options.coreml_conditioner_path = "models/minimax/conditioner.mlmodelc";
auto engine = tts_cpp::minimax::Engine::create(options);
```

Only the condition encoder's original three-tap convolution moves to Core ML
with `CPUAndNeuralEngine`. The original layer mix and nearest resampling run
on the CPU. LM, Depth, DiT and vocoder retain their native backend, including
Metal on Apple. This is a hybrid route, not a complete ANE implementation.
The Core ML model and IO buffers remain loaded between `generate()` calls;
its convolution weights are omitted from the native synth allocation.

`--frames` controls the exported window size, not the maximum generation
length. C++ supplies both convolution halos at each window boundary and
stitches every projected frame before resampling. The final partial window
and requests longer than one window use the same learned convolution.

An empty `coreml_conditioner_path` preserves native generation. An explicit
path fails when the build lacks Core ML support, the model cannot load, or
its IO contract is incompatible; prediction errors fail generation. There
is no silent switch to the native projection. Core ML controls placement
within CPU + ANE; Instruments' Core AI / ANE intervals verify hardware use.
The [memory-fit preflight](memory-fit.md#minimax-music3) reports
`coreml-memory-estimate-unavailable` for this route.

## Core ML VAE decoder sidecar

`AUDIOGEN_COREML=ON` is Apple-only. It enables an optional Core ML sidecar for
the ACE-Step Oobleck VAE decoder — the stage that dominates generation time at
song lengths on Apple GPUs — running it predominantly on the Neural Engine.
Every other stage stays on ggml, and the decoder falls back to ggml on any
sidecar failure. Export it from the VAE GGUF:

```bash
python3.11 -m venv .venv-coreml
. .venv-coreml/bin/activate
python -m pip install -r engines/parakeet/scripts/requirements-coreml.txt
python engines/audiogen/scripts/export-vae-coreml.py \
  --gguf models/vae-BF16.gguf \
  --out models/vae-decoder.mlpackage --compile-dir models
```

The compiled sidecar must sit next to the VAE GGUF as
`<basename-minus-quant>-decoder.mlmodelc` (`vae-BF16.gguf` resolves to
`vae-decoder.mlmodelc`). The export is fixed-shape; the engine decodes in
fixed overlapped windows of exactly the exported latent length (the last
window end-aligned), keeping the same trimmed-core stitching as the ggml
chunked decode. A latent shorter than one window falls back to ggml.

The default 64-frame window is the Neural Engine sweet spot, not a memory
compromise: the ANE caps the decoder's output width near 131072 samples
(68 x 1920 still fits, 72 x 1920 splits ops onto the GPU), a 68-frame window
runs over 2x slower per frame than the 64-aligned one, and a window of 128 or
more fails ANE compilation outright, leaving a Core ML GPU model an order of
magnitude slower than ggml Metal. Windows overlap by a fixed 8 frames:
stitching is bit-exact against a full decode down to a 12-frame overlap (the
decoder's receptive field) and at 8 the boundary error stays below the
sidecar's own fp16 conversion error. `--palettize {4,6,8}` additionally
compresses the weights to a k-means LUT for smaller sidecars; the parity test
is the quality gate for such exports (measured on the ACE-Step VAE: 8-bit
passes at cosine 0.9997 and halves the sidecar to 81 MB at unchanged speed,
6-bit fails the 0.999 gate). The very first load of a new sidecar pays a
one-time on-device ANE compilation (tens of seconds); the OS caches the
result for subsequent loads.

The export replaces the two `ConvTranspose1d(kernel 8, stride 4)` upsample
stages with an exact phase-convolution + depth-to-space form: the native
operation miscomputes on the Apple Neural Engine (measured on an M5, macOS 26:
output cosine 0.71 vs CPU at any size), while the phase form matches the CPU
reference at fp16 rounding error. `test-vae-coreml-parity` gates the sidecar
against the ggml decode at cosine 0.999 (measured 0.99999).

Set `ACESTEP_COREML_DISABLE=1` to force the ggml decode, including for parity
or benchmarking. `ACESTEP_COREML_STRICT=1` turns the silent ggml fallback into
a decode failure, so a test cannot measure ggml and attribute it to Core ML —
the parity test and benchmark set it for their Core ML legs.
`ACESTEP_COREML_COMPUTE_UNITS=cpu_only|cpu_and_gpu|cpu_and_ane`
overrides the default all-units placement for comparisons — on an M5 both
non-default choices are large regressions (`cpu_and_ane` pushes the
ANE-declined upsample stages onto the CPU).

`bench-vae-coreml` times the ggml GPU decode against the sidecar on the same
deterministic 30 s and 60 s latents (median of 3 after a warm-up that absorbs
the one-time ANE compile) and prints a markdown table; it fails below the
0.999 parity gate, so its numbers are correctness-checked. The audiogen CI
macOS lane runs it for the float16 and the 8-bit palettized sidecar and
appends both tables to the job summary. Measured on an M5 (macOS 26): 1.24x
over ggml Metal at 30 s, 1.32x at 60 s. In a full `music-cli` generation the
VAE stage drops from 2.6 s (Metal) to 2.2 s at 30 s and from 5.0 s to 4.0 s
at 60 s, and the decode leaves the GPU entirely.
