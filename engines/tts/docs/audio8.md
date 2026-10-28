# tts engine: Audio8

Part of the [tts engine documentation](../README.md).

## Audio8

[Audio8](https://github.com/Audio8-AI/Audio8_TTS) is a DualAR zero-shot TTS
model: a 24-layer Qwen2.5-0.6B-shaped **slow AR** emits one semantic token per
codec frame, a 4-layer **fast AR** expands each of those into the frame's
remaining 9 codebooks, and a DAC-style residual-vector-quantised **codec**
turns the 10-codebook frames into 44.1 kHz audio at 2048 samples (21.5 Hz) per
frame.  Cloning needs no speaker encoder: the codec *encoder* turns a reference
wav into codes, and those codes plus the reference transcript are prepended to
the prompt.

**Status — CPU, Metal, OpenCL, and desktop Vulkan, validated against the
reference.**  Text-to-speech and voice cloning both run in-process on macOS and
iOS Metal, on Android/Adreno OpenCL, and on Linux and Windows Vulkan.  Pass
`--n-gpu-layers 99` to offload every stage; omit it for CPU.  With several
Vulkan adapters the engine takes the one with the most free memory and never an
integrated one while a discrete one is visible: on a desktop whose Ryzen
9950X3D iGPU enumerates ahead of an RTX 5090, taking the first adapter ran
Audio8 on the iGPU, several times slower and outside the GPU accuracy bars.  On Metal that is
**4.2x** CPU at q4_0 and **3.2x** at q8_0 on an iPhone 17, measured on device
with both arms in one launch.  At F32 the GPU reproduces the CPU code trajectory
exactly, frame for frame, which is the strongest available statement that the
graphs compute the same function; quantised tiers fork their trajectories under
either backend and are judged by WER instead.

On OpenCL the two halves of the model behave differently and are reported
separately, because a single overall number is not reproducible on that device.
**Codec synthesis is 3.5-4.7x faster than CPU** and is stable run to run.  The
autoregressive loop is not GPU-bound at all: it is bound by the host issuing
roughly one dispatch every 24 us, so its wall time follows which CPU core the
issuing thread is scheduled onto.  Pinning that thread to the big cluster took a
q4_0 utterance from 30952/22011 ms across two runs to 17197/17201 ms — a 41%
spread collapsing to 0.02% — while codec synthesis did not move.

| backend | device | tier | codec synth | overall, same cores | overall, unpinned |
|---|---|---|---:|---:|---:|
| Metal | iPhone 17 | q8_0 | — | 3.2x | — |
| Metal | iPhone 17 | q4_0 | — | 4.2x | — |
| OpenCL | Adreno 740 | q8_0 | 4.7x | 1.9x | 1.1x |
| OpenCL | Adreno 740 | q4_0 | 5.1x | 2.3x | 1.3x |

Read the two overall columns as a range, not as one number with a caveat.  "Same
cores" pins both arms to the big cluster, which makes the GPU arm reproducible
(2.9% spread) but costs the CPU arm, whose five threads then contend for four
cores.  "Unpinned" is the median of every interleaved, thermally gated run, and
is what an application that does not set affinity should expect; its GPU arm
ranged 126-245 ms/frame at q8_0 for the same binary.  Both were measured on a
cold device with a 50 °C gate between arms.  Real-time factors are quoted
elsewhere in this file as wall-clock seconds per second of audio, so **lower is
faster** and a value above 1 means slower than real time.

### Convert

Three GGUFs, because the halves have different lifetimes: the LM and the codec
decoder are needed for every synthesis, the codec encoder only to enrol a voice
from a wav.  Both codec halves carry the codebooks, so each file stands alone.
`scripts/download-audio8-checkpoint.sh` fetches the upstream checkpoint from
[Audio8/Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)
(Hugging Face CLI: `pip install -U huggingface_hub`).

```bash
scripts/download-audio8-checkpoint.sh --dir models/Audio8-TTS-Preview-0.6b

python3 scripts/convert-audio8-lm-to-gguf.py \
    --model-dir models/Audio8-TTS-Preview-0.6b \
    --outfile models/audio8-lm-q8_0.gguf --dtype q8_0

for part in encoder decoder; do
  python3 scripts/convert-audio8-codec-to-gguf.py \
      --model-dir models/Audio8-TTS-Preview-0.6b --part $part \
      --outfile models/audio8-codec-$part-q8_0.gguf --dtype q8_0
done
```

| `--dtype` | LM | codec decoder | codec encoder |
|---|---:|---:|---:|
| `f32` | 2.3 GB | 499 MB | 793 MB |
| `f16` | 1.2 GB | 251 MB | 398 MB |
| `q8_0` | 801 MB | 201 MB | 252 MB |
| `q4_0` | 602 MB | — | — |

**The f16 LM is free.**  The LM checkpoint ships in bfloat16, whose 8 mantissa
bits fit inside f16's 10, so half precision stores it without loss — f32 and
f16 builds agree to 3e-8, which is below the f32 rounding of the conversion
itself.  Prefer f16 over f32 unless you are debugging.  The codec ships in f32
and its f16 build is genuinely lossy (worst tensor deviation 1.6e-3).

`--dtype` sets a ceiling, not a blanket cast.  Each tensor is classified by
what it is used for, in `role_of()` in either converter: body matmuls take the
block format, bulk weights that must stay element-addressable (embedding
tables, convolution kernels) stop at f16, and tensors that decide a greedy
choice stay f32 in every build — the RoPE tables, the sampling heads, the
codebooks (the encoder picks codes by nearest neighbour, so rounding them moves
the argmin) and every 1-D parameter, the snake alphas among them.  A q4_0 codec
is not offered: over half the decoder is convolution kernels that block formats
cannot address, so the tier would buy little and cost audible quality.

Two things the converters do that are worth knowing before reading them:

- **RoPE tables are baked, not recomputed.**  The reference builds them in
  bfloat16 even under float32 inference, and that ~2e-3 rounding is not
  cosmetic — recomputing in float32 changes the greedy fast-AR codebook choices
  on the very first frame and the trajectories separate from there.
  `--rope-precision f32` is available to reproduce that measurement.  The
  tables are emitted as separate `_cos` and `_sin` planes, the shape
  `apply_rope_in_graph` consumes.
- **Q and K rows are reordered.**  The reference rotates each head's adjacent
  pair `(2j, 2j+1)`; ggml and the rest of this tree rotate `(j, head_dim/2+j)`.
  Reordering the projection rows at conversion time avoids a strided gather in
  every attention block, and attention cannot tell the difference because the
  same permutation lands on Q and K and their dot product does not depend on
  the order of the terms.  `verify-audio8-conversion.py` checks this by
  comparing attention scores, not tensors.
- **The text head is replaced by a semantic head.**  The head is tied to the
  155776-row input embedding, but the sampler masks every logit outside
  `[semantic_begin, semantic_end]` plus EOS, so the converter emits just those
  4097 rows as `lm/sem_head`.  Masked-out logits contribute nothing to the
  top-k/top-p renormalisation, so this is exact, and it turns a 155776-row
  projection per step into a 4097-row one.

### Verify

`verify-audio8-conversion.py` reloads the checkpoint through its own remote
code and compares every converted tensor against it, at a tolerance derived
from what the tensor's storage can actually represent — exact for f32, one
reconstruction step for the block formats.  It recomputes the RoPE tables
independently rather than trusting the converter's copy.

Each GGUF is also checked for completeness on its own, against the half of the
checkpoint the converter's own split says it carries, so a tensor written to
the wrong half fails here rather than at load time.  It takes any subset of the
three files, but at least one.

```bash
python3 scripts/verify-audio8-conversion.py \
    --model-dir models/Audio8-TTS-Preview-0.6b \
    --lm-gguf models/audio8-lm-q8_0.gguf \
    --codec-encoder-gguf models/audio8-codec-encoder-q8_0.gguf \
    --codec-decoder-gguf models/audio8-codec-decoder-q8_0.gguf
```

### Reference fixtures

Three dumpers write the `.npy` fixtures the C++ stages will be checked against.
They capture stage boundaries, not just endpoints, so a failing stage can be
located without bisecting the network.

```bash
python3 scripts/dump-audio8-tokenizer-reference.py \
    --model-dir models/Audio8-TTS-Preview-0.6b --out-dir artifacts/audio8-ref

# reference-free generation: prompt, embeddings, per-step hidden states,
# semantic logits before and after filtering, fast-AR logits, emitted codes.
# --dump-wav also decodes those codes, which test-audio8-engine compares against
python3 scripts/dump-audio8-lm-reference.py \
    --model-dir models/Audio8-TTS-Preview-0.6b \
    --out-dir artifacts/audio8-ref --max-new-tokens 32 --dump-wav

# codec both ways: decode the codes above, encode a reference wav.  This is the
# recording the cloning fixtures below are enrolled from, so the tests expect it
python3 scripts/dump-audio8-codec-reference.py \
    --model-dir models/Audio8-TTS-Preview-0.6b \
    --codes artifacts/audio8-ref/codes.npy \
    --audio test/reference-audio/audio8-reference-en.wav \
    --out-dir artifacts/audio8-ref

# the cloning path end to end: reference codes + transcript -> prompt -> codes
python3 scripts/dump-audio8-lm-reference.py \
    --model-dir models/Audio8-TTS-Preview-0.6b \
    --text "Tear in eye, your dress you'll tear." \
    --reference-codes artifacts/audio8-ref/codec_enc_codes.npy \
    --reference-text "What these pictures? An elephant with uh so many legs." \
    --out-dir artifacts/audio8-ref-clone --max-new-tokens 24 --dump-wav
```

The dumps are taken with `do_sample=False`.  The shipped sampler draws Gumbel
noise and applies repetition-aware resampling, neither of which reproduces
across runtimes; greedy decoding makes the trajectory comparable, and the
filtered score vectors are dumped alongside the tokens so everything feeding
the draw can be checked even where the draw itself cannot.  `--dump-wav` runs
the generated codes back through the codec so the engine test has an
end-to-end waveform to compare against, not just codes.

CTest looks for the results in fixed places, and a fixture it cannot find turns
its test into a "Not Run (Disabled)" rather than a failure, so the recipe above
writes where the tests read: the dumps into `artifacts/audio8-ref` and
`artifacts/audio8-ref-clone`, the GGUFs into `models/`.  `cmake` prints a
`disabled (missing fixture(s): ...)` line at configure time for whatever is
still absent, which is the quickest way to see that a run is thinner than it
looks.

### Run

```bash
# text only: the LM and the codec's synthesis half
audio8-cli --lm models/audio8-lm-q8_0.gguf \
           --codec-decoder models/audio8-codec-decoder-q8_0.gguf \
           --text "Hello from a fully on-device C++ pipeline." \
           --out out.wav --threads 8 --n-gpu-layers 99

# on the GPU: same command, everything on the device
audio8-cli --lm models/audio8-lm-q8_0.gguf \
           --codec-decoder models/audio8-codec-decoder-q8_0.gguf \
           --text "Hello from a fully on-device C++ pipeline." \
           --out out.wav --n-gpu-layers 99

# cloning: add the analysis half, a reference wav and what it says
audio8-cli --lm models/audio8-lm-q8_0.gguf \
           --codec-decoder models/audio8-codec-decoder-q8_0.gguf \
           --codec-encoder models/audio8-codec-encoder-q8_0.gguf \
           --ref-audio voice.wav --ref-text "What the recording says." \
           --text "Now say this instead." --out out.wav --threads 8
```

`--greedy` reproduces the fixtures; otherwise `--seed`, `--temperature`,
`--top-k` and `--top-p` drive the sampler.  `--max-frames` caps the output at
2048 samples each (~46 ms), `--output-sample-rate` resamples away from the
codec's native 44.1 kHz, and `--backends-dir` points a `GGML_BACKEND_DL` build
at its backend libraries.  The reference wav is resampled and downmixed on the
way in, so any format `dr_wav` reads will do.

`--n-gpu-layers` is all-or-nothing by design: it selects the backend the whole
model runs on rather than splitting layers across two, so any count above zero
puts everything on the device and zero keeps it on CPU.  A GPU run copies the
weights into device memory rather than mapping them, so plan for the GGUF size
again on top of the file itself.  `--verbose` prints the per-stage times and the
block width the codec settled on, and `--dump-codes` writes the discrete code
trajectory as one comma-separated line per frame, which two runs can be diffed
on to see whether a backend or a quantisation tier changed what was generated.

Codec synthesis runs in blocks whose width is chosen from a memory budget rather
than fixed: the widest block whose scratch fits a quarter of what the backend
reports free, capped at 384 MiB.  A backend that reports no total at all is
saying it cannot tell rather than that it has nothing, and gets the cap.  Blocks
exist only to bound that scratch — each one re-runs the causal context of the one
before it — so a wider block is strictly less work, and the width never changes
the samples.  Set `codec_model::synthesis_block_frames` to pin a width or
`synthesis_scratch_budget` to pin the budget.

The same thing through the library, from `<tts-cpp/audio8/engine.h>`:

```cpp
tts_cpp::audio8::EngineOptions opts;
opts.lm_gguf_path            = "models/audio8-lm-q8_0.gguf";
opts.codec_decoder_gguf_path = "models/audio8-codec-decoder-q8_0.gguf";
opts.codec_encoder_gguf_path = "models/audio8-codec-encoder-q8_0.gguf";  // cloning only

tts_cpp::audio8::Engine engine(opts);
auto plain = engine.synthesize("Hello.");

auto voice = tts_cpp::audio8::load_voice_prompt("voice.wav",
                                                "What the recording says.");
auto cloned = engine.synthesize("Now say this instead.", voice);
```

`VoicePrompt` is mono float32 plus a transcript, so a caller that already has
samples can fill it directly; `load_voice_prompt` is there for the common case
of a file on disk.

The engine holds the models resident across calls and caches the codes for the
most recent voice prompt, so re-using a reference skips the codec encoder.
`cancel()` stops an in-flight `synthesize()` at the next language model step or
codec block, whichever comes first; the call throws rather than returning a
partial waveform.

The three GGUFs have to come from one checkpoint, and the engine checks that
they do — codebook counts and widths, the codec's two halves against each other
and both against the language model — from their headers, before it reads a
byte of weights.

### Hexagon NPU (Snapdragon)

Audio8 runs end-to-end on the Hexagon HTP0 accelerator on Snapdragon 8 Elite
devices with the ggml-hexagon backend built in. Select it with
`--backend hexagon` (CLI) or `EngineOptions::backend = "hexagon"` (API);
`--n-gpu-layers` is ignored on this path.

Configuration used for the corrected October 8 baseline:

- `q8_0` quantisation on the LM and codec decoder.
- `--greedy` decoding for repeatable output. This removes random sampling;
  it does not establish numerical parity with CPU or OpenCL.

The engine ships the Hex kernels Audio8 needs (ARGMAX among them). FastRPC
has to find the DSP library at runtime:

```sh
export LD_LIBRARY_PATH=./native
export ADSP_LIBRARY_PATH=./native   # path to libggml-htp-v79.so
```

Use these flags together for full computation with polling and fusion:

```sh
export GGML_HEXAGON_OPPOLL=1   # busy-poll DSP completion; default with ggml #115
export GGML_HEXAGON_OPSTAGE=3  # QUEUE (1) | COMPUTE (2); default
export GGML_HEXAGON_OPFUSION=1 # enable supported fusion; default
```

`OPSTAGE=1` is a profiling control that skips computation in kernels honoring
that flag. The earlier 1.87× tuning claim used this setting and is invalid for
inference, even with `--greedy`. Companion ggml #115 enables polling and the
F32-preserving DDR codec panel by default, alongside `OPSTAGE=3`, fusion and
canonical host buffers. These environment settings explicitly reproduce the
defaults. Set `GGML_HEXAGON_OPPOLL=0` or `GGML_HEXAGON_F16_F32_PANEL=0` to opt
out or for diagnostic comparisons; polling's effect on power has not been measured.

A fresh three-frame greedy check with `OPSTAGE=1` produced silent WAVs and
repeated degenerate codes with fusion both enabled and disabled. With
`OPSTAGE=3`, all four polling/fusion combinations produced identical code and
WAV files. These smoke checks isolate the flag behavior; they do not validate
speech quality or cross-backend parity.

With canonical Hexagon host-buffer handling fixed in
[ggml #115](https://github.com/tetherto/qvac-ext-ggml/pull/115), the corrected
historical same-device QDC Snapdragon 8 Elite S1 baseline uses "The quick brown
fox jumps over the lazy dog.", four threads, a 70-frame cap, three warmups and
five timed runs per variant, interleaved:

| Backend | Median inference | Inference RTF | Generated frames |
| --- | ---: | ---: | ---: |
| Hexagon, `OPPOLL=0` | 24.5906 s | 8.0229 | 66 |
| Hexagon, `OPPOLL=1` | 19.4765 s | 6.3544 | 66 |
| OpenCL | 3.3727 s | 1.0375 | 70 |

Inference time excludes model loading. Polling gives **1.2626×** faster
inference with identical code and WAV hashes across all ten timed Hexagon
runs. OpenCL generates different codes and lengths, so these measurements
do not establish cross-backend correctness or equal generated workloads.
This is one prompt, not a five-prompt validation or an audio-quality gate.

Current codec convolution matmuls request F32 precision and run on HVX. In a
separate corrected S3 profile, three large codec matmul groups account for
about 82% of leaf DSP cycles. `OPBATCH` includes operation execution; its time
cannot be treated as pure dispatch overhead or added to leaf operation times.
Further routing, precision, or weight-placement changes need separate
correctness and performance validation. Reproduction and artifact provenance
are recorded in ggml's `docs/hexagon-audio8-profile.md`.

The subsequent validated DDR-panel comparison reduced S1 inference from
20.346 s to 14.603 s median (1.393x), with identical 66-frame codes and passing
waveform gates. Those are same-device panel-off/panel-on measurements from a
separate session; they must not be combined with the table above into a new
cross-backend comparison. The later three-backend S2–S5 sweep was cancelled;
its partial results do not replace this S1 evidence. Current Hexagon-only
dispatch measurements are documented separately in the companion ggml
`docs/hexagon-audio8-profile.md` report.

#### Dispatch profiling

Companion ggml #115 copies only the live operations into each outstanding
batch's host metadata cache. It retains those snapshots for both profiling
and DSP-error diagnostics, including when queue slots are reused. This change
does not alter sampling, tensor precision, kernel routing or synchronization.
Package the matching host library and DSP skeleton from the same build.

For a diagnostic capture, use the normal greedy S1 invocation with
`GGML_HEXAGON_PROFILE=3 GGML_HEXAGON_OPTRACE=65536` and `--verbose`, saving
stdout and stderr. In the ggml checkout, run:

```sh
python3 scripts/hexagon-dispatch-profile.py audio8-s1-trace.log
```

The parser separates batch tensor preparation and worker wake/suspend from
per-operation setup, execution and retirement. Setup includes cache coherence;
execution includes nested kernel activity. Check `complete_trace` before
interpreting stage totals: saturated buffers, missing pairs or missing
operation indices make coverage incomplete. Nested cache-flush and worker
spans are not additive elapsed time. Host wait includes DSP execution and
transport, pop includes logging, and queued-batch lifetimes overlap.

Use separate runs with profiling disabled for performance claims. Greedy
decoding removes sampling randomness but does not guarantee equal CPU,
OpenCL and Hexagon trajectories. Check generated codes and frame counts,
then compare fixed-input numerical boundaries and waveforms.

#### Quantized LM projection repacking

The Hexagon backend needs repacked Q8_0 weights to execute transformer
projections on the DSP. Audio8 previously allocated these weights in canonical
host buffers, which sent the projections to the scheduler's CPU fallback.
The default loader path puts eligible slow and fast transformer projections in
Hexagon repack buffers. Set the environment variable to `0` to select the
previous canonical layout for diagnostic comparisons:

```sh
export AUDIO8_LM_HEXAGON_REPACK=0
```

Use this with `--backend hexagon` and canonical host buffers
(`GGML_HEXAGON_HOSTBUF=1`, the default). Embeddings, norms, other weights,
KV caches, and codec allocations retain their existing buffer types.
Both fused and unfused Q8_0 projections are supported; tensor uploads remain
whole-tensor operations, and memory estimation includes both weight layouts.
Leave the variable unset, or set it to `1`, for the default repacked path.

Repacking does not change the stored weights or requested matmul precision.
Projection-level CPU/OpenCL comparisons pass. Full-LM trajectories remain
sampling-sensitive and are not bit-exact across CPU, OpenCL, and Hexagon; the
validated native outputs are stable and intelligible, and the strict numerical
differences remain recorded in the performance report. Set the variable to `0`
when exact comparison with the previous Hexagon allocation path is required.

#### F32 flash attention for Hexagon decoding

Maskless F32 flash attention is enabled by default for the slow and fast LM
decoders on supported Hexagon backends. No environment setting is required.
To use separate attention operations instead, set this before loading the model:

```sh
export AUDIO8_LM_HEXAGON_FLASH_ATTN=0
```

The default path requires a Hexagon backend that supports the F32 flash-attention
operation for the model's head dimensions. The loader checks support and
selects the cache layout once. Leave the variable unset, or set it to `1`, for
the default flash path. Unsupported operations use separate attention.
CPU and OpenCL retain their existing path.
Use the companion backend with the `hvx-f32-decode` kernel for the measured
performance; the support probe checks operation support and does not identify
which kernel an older backend selects.

The enabled path stores V in position-major order, with each head's dimensions
contiguous, like K. Single-token decoding combines QK, softmax and the value
reduction in one F32 operation, keeping scores and accumulation in VTCM.
Masked prefill retains causal attention and materializes a channel-major view
of V. Cache capacity and payload size are unchanged.

The companion Hexagon kernel also vectorizes short, irregular F32 softmax
rows, including masks, reductions and normalization. This kernel is selected
automatically when applicable; it has no Audio8 CLI switch.

`test-audio8-attention-cache [CPU|HTP0]` compares the production graph with
canonical CPU attention. It covers masked prefill, decoding, grouped query
heads, layer offsets, cache reset and prefixes crossing a 128-key tile.
Full LM comparisons use fixed CPU histories because changes in F32 reduction
order can change close argmax decisions. The default flash path does not
promise identical native codes across implementations.

See [validation and performance](audio8-hexagon-attention.md) for the complete
S1-S5 inference measurements, CPU comparisons and remaining execution costs.

#### Codec panel selection

Companion [ggml PR #119](https://github.com/tetherto/qvac-ext-ggml/pull/119)
sets the F32-preserving 4x2 HVX panel as the default for large codec DDR matmuls.
Set `GGML_HEXAGON_F16_F32_PANEL_SHAPE=2x2` for the older panel or `4x2` to
select the current default explicitly. VTCM routing and precision guards are
preserved. Profiling reports `hvx-panel-4x2`; keep profiling disabled for
performance comparisons.

Fixed-code codec checks on the phone covered 3, 66, and 94 frames against
CPU/OpenCL. The blocked reuse and residual fusion experiments remain opt-in;
the four-variant prompt sweep did not show a repeatable full-inference gain.
The LM projection repacking experiment above controls a separate weight
layout and retains the codec panel setting.

#### Hexagon codec synthesis precision

The companion Hexagon backend also exposes an experimental bounded F32
convolution tile. Disable the default FP16 mode and enable the tile before
loading the decoder:

```sh
export AUDIO8_CODEC_HEXAGON_FP16=0
export AUDIO8_CODEC_HEXAGON_CONV_TILE=1
```

Eligible multi-tap convolutions whose ordinary products use the DDR 4x2
panel accumulate 64 output rows per worker in VTCM, then copy the completed
tile once. Each tap preserves the existing F32 product reduction and addition
boundary. Efficient existing VTCM products and unsupported shapes retain the
separate-tap graph. The optional registry procedure is
`ggml_backend_hexagon_conv_taps_f32`; an older backend without that procedure
also retains the separate-tap graph.

Hexagon decoder synthesis uses FP16 activation staging through the existing
HMX matmul kernels by default. No environment setting is required. Set this
before loading the decoder to restore the F32 synthesis graph:

```sh
export AUDIO8_CODEC_HEXAGON_FP16=0
```

This applies only to large F16-weight matmuls in decoder synthesis: at least
64 input and output channels, each divisible by 32. These products request
default backend precision; the backend chooses HMX when its remaining shape
and scratch requirements are met. Activations and HMX output tiles are F16
internally, with F32 graph outputs. This changes numerical precision. Small
channel tails, the latent transformer, and the LM retain their existing
precision. CPU, OpenCL, and encoder graphs retain their existing behavior.
Existing HMX matmul/ADD fusion remains available with normal backend fusion
enabled; the older experimental HVX reuse/residual-fusion switches are separate.

See the [codec experiment report](audio8-hexagon-codec-tracks.md) for the
optimized-only stage measurements, focused waveform checks and saved OpenCL
comparison.

Leave `AUDIO8_CODEC_HEXAGON_FP16` unset, or set it to `1`, for the default HMX
mode. F32 tiling remains opt-in and applies only when FP16 is disabled.
To restore the original separate-tap F32 graph, set FP16 to `0` and leave
`AUDIO8_CODEC_HEXAGON_CONV_TILE` unset or set it to `0`. LM repacking,
supported flash decoding, vectorized softmax and the 4x2 panel retain their
defaults. Keep profiling disabled for timing comparisons. The recorded
Plan B measurements used explicit FP16 enablement; this default change selects
the same synthesis graph and arithmetic.

### Core ML codec sidecar

`TTS_CPP_COREML=ON` is Apple-only. It enables an optional Core ML sidecar for
the codec's **synthesis stack** -- the two upsampling stages and the DAC
decoder, everything from the windowed post transformer's output to the
waveform. That stack is the single largest stage of a CPU synthesis (5.4 s of
a 14 s run for 4.5 s of audio on an M2) and about a quarter of a Metal one,
and it is the part of Audio8 with a fixed, convolutional shape; the
autoregressive language model, the quantizer banks and the post transformer
stay on ggml. Export the sidecar from the decoder GGUF:

```bash
python3.11 -m venv .venv-coreml
. .venv-coreml/bin/activate
python -m pip install -r engines/parakeet/scripts/requirements-coreml.txt
python engines/tts/scripts/export-audio8-codec-coreml.py \
  --gguf models/audio8-codec-decoder-q8_0.gguf --compile-dir models
```

The compiled sidecar must sit next to the decoder GGUF as
`<basename-minus-quant>.mlmodelc`: `audio8-codec-decoder-q8_0.gguf`,
`-f16.gguf` and `-f32.gguf` all resolve to `audio8-codec-decoder.mlmodelc`,
because every tier stores the synthesis stack's kernels at f16 or better and
the export is the same whichever file it was taken from. The exporter
rebuilds the stack in PyTorch from the GGUF tensors, so it needs neither the
checkpoint nor its remote code; `--parity-dir artifacts/audio8-ref` checks
that rebuild against the reference dumps first (measured: waveform cosine
1.0000000, max error 1e-6). With the sidecar present `audio8-cli --verbose`
reports `codec synthesis on the Core ML sidecar` at load and the compute
label (`coreml-all`) in the timing breakdown.

The export is fixed-shape, `--window` post frames (default 64, 2.97 s,
131072 samples) in and the matching samples out. The engine walks an
utterance in windows of exactly that width: every convolution in the stack is
causal, so a window's output is exact from the first frame whose receptive
field lies inside it, and the engine drops each window's leading causal
context (10 frames for this checkpoint, computed by the same walk the ggml
block path uses) just as the ggml blocks drop theirs. An utterance shorter
than one window is zero-padded on the right, which changes nothing before the
padding because nothing in the stack looks forward. `test-audio8-coreml-windows`
locks that plan without a model; `test-audio8-codec-coreml-parity` gates the
sidecar against the ggml synthesis at cosine 0.999 on a short (padded) and a
long (stitched) utterance, and checks that a cancel between windows stops the
pass; both run on the TTS CI macOS lane, which converts the decoder from the
upstream checkpoint and exports the sidecar itself.

**The sidecar works while the language model is still generating.**
`Engine::synthesize` hands each window to a worker thread as soon as the frames
it covers exist, so on anything longer than one window most of the codec runs
alongside the autoregressive loop and only the last window is left once
generation stops.  Every window but the end-aligned last one sits where it
would sit in the plan for any longer utterance, so the stream hands off exactly
the windows the after-generation pass would pick, each as soon as one frame
past its end exists, which is what proves it is not the last.  Its input comes
from the post transformer run a chunk at a time: the transformer is causal
with a 128-frame attention window per layer, so each chunk computes only its
new frames, attending over the rotated keys and the values the previous chunk
left for its last 127 positions in every layer, and an utterance costs one
post-transformer pass however it is cut (recomputing each window's history
instead cost 190 ms per 512 frames on an M4 mini and 600 ms on a 42 s
utterance, against 30-75 ms for one pass).  `test-audio8-codec` checks that
one chunk is the whole-sequence pass exactly and that a run cut into chunks of
1 to 200 frames matches it (9.5e-7 on the CPU), and fails when the history
held is one position short.  Those passes are ggml graphs, so they run on the
calling thread between language-model steps (ggml-metal shares one command
queue per device); only the sidecar predictions run on the worker.  A cancel
stops the worker between windows, and a failed prediction retires the sidecar
and reruns the whole utterance on the ggml blocks after generation, as before.
`test-audio8-window-stream` drives the scheduler with a fake causal
synthesizer at every length up to four windows and checks that it asks for
each frame's post column once; `test-audio8-codec-coreml-parity` holds the
streamed waveform to the after-generation one (cosine 0.99999; measured
0.999996-0.999999 on an M3 Ultra and an M4 mini) and checks that every window
but the last went during generation.
`AUDIO8_COREML_STREAM_DISABLE=1` synthesises after generation instead; the
Apple silicon table under Engine notes measures both.  In
the timing breakdown `codec-latent` then includes the streamed post passes,
and `codec-synth` counts only what is left after the last frame;
`--verbose` reports how many windows went during generation.

Overlap only pays where the sidecar does not compete with the language
model, which holds the GPU and, on Metal, one host core issuing its kernels.
Measured with `audio8-cli` at all three tiers, on the 512-frame benchmark text
and on 37-42 s utterances, against the same build synthesising after
generation:

- On an M3 Ultra (macOS 15.7.9) the folded export's all-units plan puts 125
  of its 670 operations on the Neural Engine and 544 on the GPU. Streaming
  takes 0.75-1.28 s of synthesis off the end and costs the language model
  30-150 ms of GPU contention, so a run is 9-11% shorter.
  `cpu_and_ane` keeps the sidecar off the GPU, but synthesises at half the
  speed (1.73 s per 512 frames against 0.83 s), so streamed it lands within
  1% of all-units with a 170 ms tail instead of 72 ms.
- On an M4 mini (macOS 15.3.2) the plan is Neural Engine only, all-units and
  `cpu_and_ane` measure the same, and synthesis takes 2.2 s per 512 frames.
  Streaming leaves only the last window's 0.22 s, but the Neural Engine draws
  on the memory bandwidth the language model's decode is bound by, which
  slows the decode by 1.2-2.3 s, so a run is 5-8% shorter at `q4_0` and
  `q8_0` and 2-3% at `f16`, the most bandwidth-bound tier.

The CPU is no better a place for it: `cpu_only` takes performance cores the
language model's host thread needs (the M4 mini's fast-head decode grew by
0.78 s per 512 frames) and runs the f16 program less accurately (cosine
0.9992-0.9996 against 0.99999). Measured end to end the same way, the folded
export synthesises after generation more slowly than the flat `torch.sin` one
on both machines (0.84 against 0.39 s per 512 frames on the M3 Ultra, 2.23
against 1.57 s on the M4 mini), unlike on the M2 below; streamed, the M3
Ultra's difference disappears into generation, while the M4 mini's run is
0.45 s longer with it.

Four things the exporter does are worth knowing before reading it, all of
them there to keep the whole stack on the Neural Engine, whose compute plan
was read with `MLComputePlan` on an M2 running macOS 15.7. Every transposed
convolution is emitted in its exact phase form -- a causal `Conv1d` over the
input followed by a depth-to-space shuffle, which is also how
`codec_ops.cpp` computes it -- rather than as `ConvTranspose1d`, whose native
Neural Engine kernel miscomputes at stride 4 (the ACE-Step VAE export hit
this first). The Snake activation's sine is a range-reduced polynomial, not
`torch.sin`: the Neural Engine has no `sin` (or `cos`) kernel, so every one of
the 29 Snakes would otherwise bounce to the CPU. The argument is reduced in
*turns* (u = t / 2pi, f = u - round(u), r = 2pi f) and sin(r) is a degree-9
odd least-squares fit on [-pi, pi] (max error 6e-6); reducing in radians,
t - 2pi round(t / 2pi), is miscomputed on the Neural Engine (NaN and inf on
the M2) while the turns form is exact, and the stack's activations keep
|alpha x| below 7 anyway. The square is spelled as a product, never `** 2`:
the `pow` the latter lowers to is miscomputed on the Neural Engine when its
result feeds a convolution (the first DAC stage came out at cosine 0.39
against the CPU). And every stage longer than 8192 samples is folded into
rows of 8192, (1, C, L) -> (1, C, L / 8192, 8192): the Neural Engine takes
this stack's causal convolutions only up to 16384 samples per row, and from
32768 on the plan drops them to the GPU and then alternates devices op by op
through the whole tail, which is where the original export lost most of its
speed. Each causal convolution takes its left context from the end of the
previous row (zeros before the first row, which is the causal padding), the
transposed convolution's depth-to-space runs along the row and its output is
re-split into rows of the same width, so the fold is exact: the PyTorch
rebuild matches the reference decode at cosine 1.0000000 with it, and the
compiled plan places 669 of its 670 operations on the Neural Engine (the
last is the output reshape). `--snake sin` and `--fold-rows 0` export the
flat, `torch.sin` form for comparison.

Measured on an Apple M2 (macOS 15.7, f32 decoder GGUF, ggml Metal as the
reference, `bench-audio8-codec-coreml`, median of 3 inside each cell),
synthesis stage only, window 64, all-units placement. The flat `torch.sin`
export is the original sidecar; the folded polynomial export is the default.
The two were run interleaved over three rounds because a laptop's Metal
baseline drifts by up to 2x between cells as the chip heats; the Core ML
times are the stable column, so read those:

| export | 10 s (216 frames) | 24 s (517 frames) | parity |
|---|---:|---:|---:|
| ggml Metal (reference) | 1587 / 3269 / 2265 ms | 4294 / 7068 / 6082 ms | |
| flat, `torch.sin` | 1287 / 2037 / 1599 ms | 3308 / 4281 / 4104 ms | 0.99999 |
| folded, polynomial (default) | 1334 / 1380 / 1338 ms | 4526 / 3262 / 3293 ms | 0.99998 |

Against the flat export the folded one is 1.2x faster on the 10 s decode and
1.25x on the 24 s one once both are warm; against the Metal baseline it lands
between 1.2x (Metal's best cell) and 1.9x (its typical cells). The flat
export's numbers show why it needed help: with `sin` on the CPU and the tail
on the GPU its all-units plan alternated between three devices, and
`AUDIO8_COREML_COMPUTE_UNITS=cpu_and_gpu` was its faster placement (1.4x);
with the folded polynomial export the plan is Neural Engine only, all-units
and `cpu_and_ane` measure the same, and `cpu_and_gpu` (which then runs the
polynomial on the GPU) is the slower choice. A 128-frame window buys nothing
on the Neural Engine and collapses under mixed placement, so 64 stays the
default at a causal-context overhead of 10 in 64 frames. The very first load
of a fresh export pays a one-time on-device compilation (tens of seconds),
which the OS caches for later loads.

Those are steady-state numbers for the synthesis stage alone, measured before
the sidecar overlapped generation. A one-shot `audio8-cli` run also pays the
sidecar's first-prediction warm-up, which now lands on the worker during
generation for anything longer than one window. The engine keeps the sidecar
resident across `synthesize()` calls, so a host that speaks more than once
pays the load and the warm-up once.

Set `AUDIO8_COREML_DISABLE=1` to force the ggml synthesis, including for
parity or benchmarking. `AUDIO8_COREML_STRICT=1` turns the silent ggml
fallback into a synthesis failure, so a test cannot measure ggml and attribute
it to Core ML -- the parity test and benchmark set it for their Core ML legs.
`AUDIO8_COREML_COMPUTE_UNITS=cpu_only|cpu_and_gpu|cpu_and_ane` overrides the
default all-units placement for comparisons, and `AUDIO8_COREML_STREAM_DISABLE=1`
moves the synthesis back after generation. `bench-audio8-codec-coreml`
times the ggml GPU synthesis against the sidecar on deterministic 10 s and
24 s code sequences (median of 3 after a warm-up that absorbs the one-time
on-device compilation of a fresh export) and prints a markdown table; it
fails below the parity gate, so its numbers are correctness-checked, and the
TTS CI macOS lane appends its table to the job summary.

Two reports tell a host where the codec ran. `Engine::codec_on_coreml()` is
the attachment status: true while a sidecar next to the decoder GGUF is
attached (the language model and the post transformer still run on
`backend_name()`).
`SynthesisResult::codec_synthesis_backend` is per call: `"ggml"`, or the
sidecar's compute label (`coreml-all`, `coreml-gpu`, ...) when the synthesis
stack actually ran there. The two differ on the call in which a loaded
sidecar fails -- a window that cannot carry the causal context, or a Core ML
prediction failure: the engine falls back to the ggml blocks for that call and
retires the sidecar, so every later call goes straight to ggml and
`codec_on_coreml()` turns false, instead of paying a failed Core ML attempt
ahead of each full ggml pass. Because that fallback stays possible, the memory-fit projection (`audio8-fit-params`)
prices the ggml synthesis arena whether or not a sidecar is present; with a
working sidecar it over-reports by that arena rather than under-reporting the
fallback. `test-audio8-codec-coreml-parity` covers all of it: an absent
sidecar under `AUDIO8_COREML_STRICT`, a sidecar directory that is not a model
(load status false, ggml synthesis), and a sidecar exported at a window that
cannot carry the context (`--window 8`; loaded, fails under
`AUDIO8_COREML_STRICT`, falls back bit-exactly to the ggml decode and is
retired, so the next call reports ggml with no sidecar attached), each also
through the public `Engine` with a synthetic language model.

### Engine notes

Roughly a second of audio per second of CPU on eight cores of a desktop x86-64,
q8_0 weights: 16 s of speech in 19 s wall, 5.5 s of cloned speech in 10 s
including enrolment. For measured multi-machine numbers, see
[the Audio8 benchmark in Performance](performance.md#audio8-multi-machine-benchmark-2026-09).

**The codec is chunked, in both directions.**  Its convolution stacks, not the
LM, are what made memory grow with utterance length — a 24 s decode used to
need a 1.5 GB arena.  Each direction is now split into a whole-sequence graph
and a block graph that walks the utterance, and each block is re-fed the exact
receptive field its stack needs (`span_through_*` in `codec_ops.cpp` walks the
strides and dilations to compute it) and drops the samples that context
produced.  The result is bit-identical to running in one pass whatever the
block size, which `test-audio8-codec` asserts by re-running both directions at
a deliberately tiny block and comparing.  Decode now holds ~130 MB flat, encode
~140 MB for a 10 s reference.  End to end that is 1.25 GB resident for
text-only and 1.67 GB for cloning, of which 1.25 GB is the q8_0 weights
themselves.

The one part that still grows with input length is the encoder's analysis
transformer: `ggml_soft_max_ext` materialises the full score matrix, so a
reference much beyond 30 s gets expensive.  References of a few seconds are
what the model expects anyway.

**Quantised weights take a different route per backend.**  `GGML_PREC_F32`
asks a matmul to accumulate in f32, which a block-quantised weight has already
spent: on CUDA the default route is the integer dot product every CPU build of
this tier also takes, so the marker there buys nothing and costs a
dequantise-to-f32 round trip through cuBLAS: 48% of GPU kernel time, and a
fifth of the decode's wall, the loop being host-bound rather than GPU-bound.
The graphs therefore drop it on CUDA-resident quantised weights and keep it
everywhere else, because ggml-vulkan reduces quantised matmuls in f16 under
`PREC_DEFAULT`.  It reaches the codec too: its post-quantiser transformer is
56 of the q8_0 decoder's tensors.

Measured against the PyTorch fixtures at q8_0, both halves land at the CPU
q8_0 build's own accuracy, which is what this tier is judged at — the previous
CUDA route was *more* accurate than CPU, at twice the cost, because it never
quantised the activations.

| RMS deviation from the f32 reference | CPU q8_0 | CUDA before | CUDA after |
|---|--:|--:|--:|
| LM semantic logits | 0.106 | 0.077 | 0.100 |
| LM fast logits | 0.182 | 0.146 | 0.173 |
| codec latent | 2.20e-2 | 1.19e-2 | 2.18e-2 |
| codec waveform | 7.04e-4 | 3.16e-4 | 5.52e-4 |

f32 and f16 weights keep the marker on every backend, so the exact CPU/GPU
trajectory match at F32 is unaffected.  `test-audio8-quantised-precision` pins
the scoping per backend, over the LM's built graphs and the codec decoder's
resident weights.

**The fast head's graphs are built once and replayed.**  Every frame walks the
same positions with the same shapes, and a single position attends to the
whole prefix, so its causal mask would be all zeros and no mask is built at
all; the slow decode step drops its mask for the same reason, and adding
zero before the softmax was exact, so nothing changes.  A cached
graph must own its allocator (a shared arena moves under the other graphs the
first time a bigger one reserves) and must never reach the scheduler fallback,
which resets one shared arena per graph; a build that lands there drops what
was built and the per-call path serves the rest of the model's life.
`test-audio8-fast-cache-sched` pins both transitions.

**The language model's projections are fused at load.**  A decode step is one
token wide, so on a GPU its cost follows the number of dispatched kernels
rather than the bytes of weights they read: on Metal an M3 Ultra paid about
8 us per dispatch with roughly 24 dispatches per layer, 600 for a slow step
and 1100 for a frame of the fast head.  The loader therefore stacks each
layer's `wq`, `wk` and `wv` (and their biases) into one `wqkv` weight and `w1`
and `w3` into one `w13`, by concatenating rows as the file is streamed -- a row
is whole blocks in every storage type, so this works for every tier without
reconverting a GGUF, and each stacked weight is staged on the host and uploaded
in one call, which the OpenCL quantised layouts need.  The graph then runs one
QKV matmul instead of three, rotates the query and key heads together in one
four-kernel RoPE instead of two, and feeds one gate/up matmul to
`ggml_swiglu` instead of two; a one-position step also skips the head-merge
copy, whose output already has the merged layout, so it issues eight fewer
kernels per layer.
Every row is still computed by the same kernel from the same inputs, so the
output is bit-identical: `test-audio8-lm-fusion` loads a synthetic model with
distinct weights, in f32 and with q8_0 matrices, both ways and compares a
prefill, decode steps and a sampled fast-head frame byte for byte (it fails if
the rows are stacked out of order or the keys skip the rotation), and on the
real q8_0 and f32 models the codes and the waveform match the unfused build
exactly on CPU and on Metal.  `AUDIO8_LM_FUSION_DISABLE=1` loads the separate
weights for comparison.  Since the graph cuts heads and halves by the hparams,
the loader checks every projection against that geometry, stacked or not, and
refuses a file that disagrees by name instead of aborting in a reshape on the
first step; a file that already carries a tensor under a stacked name keeps
its own.  Measured end to end below; on CPU, which is not
dispatch-bound, it is neutral.

**Measured on Apple silicon (2026-10).**  One `audio8-cli` synthesis per
process, seed 42, the synthesis time `--verbose` reports (median of three after
a warm-up), on ggml `speech@d96479a1`: the 76-word benchmark text at the
default 512-frame cap (23.8 s) and a 117-word one that runs to 792-907 frames
by tier (37-42 s).  Metal is the ggml lane, master `c0151cd5` against this
change.  Core ML is the folded codec sidecar at its default all-units placement
next to the Metal language model, master `faa3f07a` synthesising after
generation against this change synthesising during it.  An M3 Ultra (60-core
GPU, macOS 15.7.9) and an M4 mini (10-core GPU, macOS 15.3.2); every Metal
waveform is byte-identical before and after on both.

| machine | LM tier | Metal, 512 frames | Core ML, 512 frames | Metal, 37-42 s | Core ML, 37-42 s |
|---|---|---:|---:|---:|---:|
| M3 Ultra | `q4_0` | 7.51 -> 5.90 s, **1.27x** | 7.83 -> 5.50 s, **1.42x** | 13.47 -> 10.59 s, **1.27x** | 13.87 -> 9.83 s, **1.41x** |
| M3 Ultra | `q8_0` | 7.66 -> 6.15 s, **1.25x** | 7.98 -> 5.85 s, **1.37x** | 12.34 -> 9.92 s, **1.24x** | 12.76 -> 9.39 s, **1.36x** |
| M3 Ultra | `f16` | 8.54 -> 6.68 s, **1.28x** | 8.85 -> 6.42 s, **1.38x** | 13.21 -> 10.52 s, **1.26x** | 13.77 -> 10.03 s, **1.37x** |
| M4 mini | `q4_0` | 10.60 -> 10.01 s, **1.06x** | 9.84 -> 8.50 s, **1.16x** | 19.14 -> 18.02 s, **1.06x** | 17.63 -> 15.18 s, **1.16x** |
| M4 mini | `q8_0` | 12.91 -> 12.31 s, **1.05x** | 12.19 -> 10.94 s, **1.11x** | 20.85 -> 19.86 s, **1.05x** | 19.46 -> 17.49 s, **1.11x** |
| M4 mini | `f16` | 17.28 -> 16.69 s, **1.04x** | 16.50 -> 15.62 s, **1.06x** | 26.89 -> 25.95 s, **1.04x** | 25.68 -> 23.95 s, **1.07x** |

The language model is most of every run (its decode steps were 94% of the M3
Ultra's), so the fused projections carry the M3 Ultra, whose large GPU leaves a
step dispatch-bound; the M4 mini is closer to its memory bandwidth and gains
4-6% from them.  The overlap adds 9-11% on the M3 Ultra and 2-8% on the M4
mini against the same build synthesising after generation (above).  With
both, the Core ML lane is 1.13-1.14x the Metal lane at `q8_0` on the M4 mini
and 1.05-1.06x on the M3 Ultra.

**Sampling follows the reference's order, which is unusual.**  top-k and top-p
run on the raw logits and the temperature is applied to what survives, so
temperature does not affect which candidates are in the running.  Semantic
tokens additionally go through repetition-aware resampling: drawing a token
that already appears in a short trailing window triggers one re-draw at a
narrower nucleus and a higher temperature.  The window holds tokens the model
actually drew and nothing else, so an utterance opens with no history at all;
the opening token stays out of it, as it does in the reference.  `--greedy`
bypasses all of it.

### Test

The C++ stages are checked against the fixtures above, one CTest target per
stage boundary:

```bash
ctest -R audio8 --output-on-failure
```

| target | checks |
|---|---|
| `test-audio8-tokenizer` | BPE ids and the ChatML prompt, both cloning and not |
| `test-audio8-lm` | prompt embeddings, per-step hidden states, semantic and fast-AR logits, emitted codes |
| `test-audio8-codec` | encode and decode at every stage boundary, block-size independence, and that a cancel stops the block loop |
| `test-audio8-sampler` | filtered score vectors, which is the part of the draw that is reproducible |
| `test-audio8-ras` | the repetition-aware window: which draws enter it, eligibility, eviction, and the retry's nucleus |
| `test-audio8-engine` | both public paths end to end against the decoded waveforms, and the refusal of GGUFs that disagree |
| `test-audio8-timing` | that the per-stage times are disjoint and bounded by the total they are reported against |
| `test-audio8-cli` | the CLI's flags, `-ngl` and `--n-gpu-layers` among them, and the `--dump-codes` file format |
| `test-audio8-cli-verbose` | the same flags through the binary: `--verbose` reaching stderr and codes surviving a synthesis |
| `test-audio8-sampling-filter` | the top-k / top-p / temperature candidate filter on synthetic score vectors: k and p cutoffs, temperature ordering, degenerate cases |
| `test-audio8-lm-fusion` | the load-time stacked projections against the separate weights, byte for byte, on a synthetic LM in f32 and q8_0 |
| `test-audio8-window-stream` | the Core ML synthesis stream with a fake causal synthesizer: which windows go during generation, stitching, failure, cancel |

`test-audio8-ras`, `test-audio8-cli`, `test-audio8-sampling-filter`,
`test-audio8-lm-fusion` and `test-audio8-window-stream` are the ones that run
on a checkout with no models; every other target needs the dumps.

On a build with a GPU backend compiled in, the `lm`, `codec` and `engine`
suites are registered a second time per backend as `test-audio8-<suite>-cuda`,
`-metal`, `-vulkan`, `-opencl` and `-webgpu`. Each arm names its backend in `AUDIO8_TEST_GPU` and
asserts it before reading a number, so an arm that fell back to CPU, or that was
handed the other arm's GPU, fails rather than passing on someone else's result.

The engine test hands the cloning path a wav rather than pre-computed codes, so
it exercises the codec encoder the way a caller would.

#### Fixed-code decoder benchmark

Both output paths must differ even under ASCII case folding; on Windows the
check also covers Unicode case aliases. This prevents two nonexistent output
names such as `out.wav` and `OUT.WAV` from overwriting each other on a
case-insensitive filesystem. Existing input/model aliases, hard links and
symlinks are also rejected. With tests and executables enabled,
`test-audio8-codec-bench-runner` generates a tiny decoder and exercises the
actual CLI, including repetition statistics and byte-level WAV/F32 exports.

`audio8-codec-bench` can decode captured codes without running the language
model. Build the existing benchmark target, then use the CSV produced by
`audio8-cli --dump-codes` (one frame per line, ten comma-separated codes):

```bash
build/audio8-codec-bench \
  --codec-decoder models/audio8-codec-decoder-q8_0.gguf \
  --codes utterance.codes --frames 66 --codec-backend hexagon \
  --warmup 3 --repeat 5 --threads 4 \
  --out decoded.wav --out-f32 decoded.f32
```

Supported decoder backends are `cpu`, `hexagon` and `opencl`. Selection is
checked; unsupported operations can still use the CPU scheduler fallback.
The report identifies the synthesis backend and fallback availability, not
the number of operations actually executed on each backend. Run with the
matching backend libraries; Hexagon correctness and timing require
`GGML_HEXAGON_OPSTAGE=3`.

Timing covers the production `decode_codes` call, including graph setup,
execution and PCM readback. Model loading, code validation/transposition and
file output are excluded. Warmup defaults to one and repeat to five; zero
warmups are allowed. Each timed iteration reports its time and decoder route.
The summary reports geometry, whether PCM is bit-identical across timed
iterations, minimum, mean, median, maximum and population standard deviation. Output files
contain the last iteration; compare F32 output as well as WAV when checking
numerical correctness.

`--codes-format i32` accepts headerless little-endian int32 codes in frame
order, with semantic codebook first and residual codebooks following.
`--codebooks N` must match the model; when omitted it is inferred. Code count,
token range, output geometry and finite PCM are validated. Existing encoder
and decoder WAV roundtrip mode remains available.

#### Fixed-input backend parity

`test-audio8-backend-parity` compares a selected `cpu`, `hexagon`, or `opencl`
backend with the CPU reference kernels using real LM and decoder GGUF files.
Build with `TTS_CPP_BUILD_TESTS=ON`, then:

```bash
cmake --build build --target test-audio8-backend-parity -j
build/test-audio8-backend-parity \
  --lm models/audio8-lm-q8_0.gguf \
  --codec models/audio8-codec-decoder-q8_0.gguf \
  --backend hexagon --frames 3 --threads 4 --text "The signal is clear."
```

The same target supports Android cross builds. Run it on the device with the
matching ggml libraries and backend support files. For Hexagon, use
`GGML_HEXAGON_OPSTAGE=3` (queue and compute); `1` only queues work and is not a
valid correctness or performance baseline. Use `--backend cpu --frames 1` to
compare CPU reference kernels against the default optimized CPU kernels.
Backend selection is verified after loading; scheduler
fallback for unsupported operations is still allowed.

Slow prefill and decode consume the same tokenized prompt and CPU-selected
codes on both models. Each fast-AR logit comparison uses the CPU's carried
input, semantic token, and preceding codebook choices. A separate check compares
the candidate's greedy per-position `fast_step` with its chained `fast_frame`.
Finally, both codecs decode the same CPU codes, comparing semantic, residual,
post, latent, and PCM outputs. LM models are released before codec loading.

Each boundary reports finite checks, cosine, normalized squared error
(`sum((test-reference)^2) / sum(reference^2)`), and maximum absolute error.
Logits also report the top token, its margin, and the CPU winner's candidate
rank; slow logits include EOS rank. Zero vectors match only when both are zero.
All available stages run even after a numerical mismatch. Exit status is `0`
only when every cosine is at least `0.9999`, normalized squared error is at most
`0.0002`, every value is finite, all compared
top tokens agree, and the chained codes agree with per-position greedy codes;
otherwise it is `1`. Near-tie token disagreements remain visible even when the
cosine gate passes. These checks cover the requested short trajectory, not
full-utterance quality or throughput.

`--precise-outputs` enables the candidate LM and codec precision flags before
graph construction for a diagnostic ablation. It changes no production
default. `--frames` defaults to three; `--frames 1` omits slow decode and is
useful for quick prefill, fast-AR, and codec checks.
CPU-selected EOS ends the trajectory early; the summary reports completed and
requested frame counts so a passing shorter run does not imply full coverage.

To locate accumulated slow-transformer error, `--slow-layers N` keeps only the
first N slow blocks in both graphs, retaining the loaded weights and KV cache
capacity. These are diagnostic truncated-model outputs, not normal generation.
Try `--frames 1 --slow-layers 1 --skip-codec`, increasing the cutoff to bracket
the divergence. `--skip-codec` also makes the decoder GGUF optional.

For CTest, set `AUDIO8_PARITY_LM`, `AUDIO8_PARITY_CODEC`, and optionally
`AUDIO8_PARITY_BACKEND` (default `cpu`), then run
`ctest --test-dir build -R '^test-audio8-backend-parity$' --output-on-failure`.
Missing or unreadable model files return `77`, which CTest records as skipped.
No exported reference fixtures are needed.

October 8 QDC Snapdragon 8 Elite check, using Q8_0 LM/decoder, four threads,
three teacher-forced frames, and "The quick brown fox jumps over the lazy dog.":

| Candidate vs CPU reference | Numeric boundaries passed | Top-token disagreements | Chained/per-step disagreements |
| --- | ---: | ---: | ---: |
| Optimized CPU | 38/38 | 0 | 0 |
| Hexagon, default host buffers | 38/38 | 0 | 0 |
| OpenCL | 26/38 | 3 | 0 |

Hexagon used `OPPOLL=1`, `OPSTAGE=3`, `OPFUSION=1`, `HOSTBUF=1` and the
canonical host-buffer fix in ggml #115. Its lowest boundary cosine was
`0.9999272188`, with maximum NMSE `0.0001491884`. Its PCM cosine for the fixed
codes was `0.9999998572`. This does not contradict differing end-to-end greedy
trajectories: the fast-AR comparisons deliberately replace the candidate's
slow hidden state with the CPU state. Small slow-state changes can alter a
later winner even when the continuous metrics pass.

OpenCL's lowest cosine was `0.9998118975` and maximum NMSE `0.001090075`.
Its failures mean OpenCL output alone is not an established correctness
reference for further Hexagon tuning. A separate `HOSTBUF=0` repacking
ablation also failed numerical and token gates; it is not an accepted tuning
recipe. These short checks do not establish full-utterance audio quality.

Device runtime source includes ggml `39b36439` and speech `0f75ca40`; the
reviewed harness is `41e7df68`. Model identities are the same as the corrected
S1 baseline in ggml's profiling document. Local raw evidence and the parsed
summary are `hexagon-26397-build/results/parity-v2.log` and
`parity-v2-summary.json`; `hexagon-26397-build/run-parity-v2.sh` records the
device invocations. These diagnostics are not performance measurements.

### Incremental PCM streaming

`Engine::synthesize_stream(text, callback)` and
`Engine::synthesize_stream(text, voice, callback)` deliver mono float PCM as
codec windows complete. The callback takes `(const float * samples, size_t count,
int sample_rate)` and returns `true` to continue or `false` to cancel. The sample
pointer is valid only during the callback. Calls are ordered and serialized,
but may run on the Core ML worker thread; do not re-enter the engine from a
callback. All callbacks finish before synthesis returns or throws.

The returned `SynthesisResult::pcm` contains the same samples as the concatenated
callbacks. Output sample-rate conversion carries sinc history across callbacks
and flushes its final samples at completion. Cancellation throws; consumers may
already have received a prefix. A codec failure after delivery starts also
fails the call instead of replaying that prefix through a batch fallback.

With a Core ML sidecar, synthesis overlaps language-model generation on the
existing codec worker. With ggml, windows run synchronously between generation
steps so codec graph allocation is never shared concurrently. ggml retains the
codec's causal left context and advances by 16 new frames per full window.
The final window is trimmed to the generated frame count. Short utterances can
finish generation before their first window is ready. Models without finite
post-transformer attention windows use batch decoding and one final callback.

For a local batch/stream comparison, run the same model, text, seed, backend,
thread count, and output rate with and without `--stream`:

```bash
./build/audio8-cli --lm models/audio8-lm-q8_0.gguf \
  --codec-decoder models/audio8-codec-decoder-q8_0.gguf \
  --text "The same benchmark text for both runs." --greedy --verbose --out batch.wav
./build/audio8-cli --lm models/audio8-lm-q8_0.gguf \
  --codec-decoder models/audio8-codec-decoder-q8_0.gguf \
  --text "The same benchmark text for both runs." --greedy --verbose --stream --out stream.wav
```

`[audio8-performance]` reports first nonempty PCM callback latency, synthesis
wall time, audio duration, real-time factor (wall seconds / audio seconds), and
callback count. Model loading and WAV writing are excluded. In batch mode,
first-audio latency is synthesis completion time. The CLI writes the complete
WAV after synthesis; its callback timing measures availability to an embedding
application, not speaker playback. These native metrics do not include SDK
text chunking or duplex input delays.

#### Local CPU measurement (2026-10-22)

On a Ryzen AI 9 HX 370 under WSL2 Linux, the draft incremental implementation
was measured with q8_0 LM/decoder weights, four CPU threads, greedy decoding,
and a fixed 51-word English prompt. Three fresh-process runs per mode used
batch/stream, stream/batch, batch/stream order. Loading and WAV writing were
excluded; inference was not warmed up. Every run emitted 426 frames (19.783 s),
below the 512-frame ceiling. The Release build used `GGML_NATIVE=OFF`.

| Metric (median of three runs) | Batch | Incremental PCM |
| --- | ---: | ---: |
| First audio available | 33.471 s | 2.172 s |
| Complete synthesis | 33.471 s | 43.613 s |
| Real-time factor (lower is better) | 1.692 | 2.205 |
| PCM callbacks | — | 26 |

First-audio latency fell 93.5%, but complete synthesis took 30.3% longer
(23.3% lower throughput). This CPU implementation does not sustain real-time
playback. Codec synthesis accounted for most of the regression: the first pair
spent 22.384 s there in batch versus 32.666 s streaming. Smaller windows repeat
causal context and add graph work, so throughput tuning remains necessary.

Generated codes and sample counts agreed across all six runs. Streamed and
batch PCM were not bit-identical: their PCM16 correlation was 0.9999981,
normalized RMSE 0.00027045, and maximum absolute difference 615 PCM16 units.
These numerical measurements are not a listening-quality assessment.

These are native CPU measurements on this machine, not a reproduction of the
SDK long-text or duplex baseline. Core ML, GPU, voice cloning, and SDK/duplex
latency were not measured in this run.

#### CI measurement

The standard TTS and cross-platform PR workflows run the fixture-free window,
codec integration, resampler, and benchmark-driver tests. The public streaming
API test is fixture-gated in CTest and also runs explicitly in the opt-in
full-model benchmark below.

Dispatch `speech-benchmark-desktop.yml` on the published feature branch with
`model_families=audio8-tts`, `audio8_pcm_streaming=true`, `runs_per_bench=3`, and
`runners=linux,macos` (or one runner). The existing workflow downloads the q8_0
models from its configured registry. Its normal Audio8 benchmark is followed
by public-API checks on CPU and paired batch/streaming runs using the runner's
available backend. The job summary reports first-audio and total synthesis
medians. The `bench-audio8-tts-<runner>` artifact contains `audio8-streaming/`
with exact commands, per-run measurements, logs, codes, and WAVs.

The paired benchmark requires matching generated codes and sample counts. It
reports timing without a speed threshold, because shared runners are variable
and CPU streaming currently trades throughput for first-audio latency. It does
not export or test a Core ML sidecar; that path still requires the macOS TTS
Core ML lane and additional full-engine measurements. SDK/duplex timing remains
a separate integration check. Dispatch must run in a repository with the
workflow's existing model-registry credentials and runner access.
