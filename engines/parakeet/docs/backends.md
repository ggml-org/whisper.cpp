# parakeet engine: backends

Part of the [parakeet engine documentation](../README.md).

## Backends and runtime selection

Any combination of Metal, CUDA, Vulkan, and OpenCL may be built or dynamically
loaded. Parakeet selects one primary GPU through the ggml backend registry when
`n_gpu_layers > 0`; this is not a fixed compile-time CUDA/Metal/Vulkan/OpenCL
cascade. `n_gpu_layers` is currently a boolean offload request: positive means
the whole encoder, not a partial layer count.

Set `EngineOptions::backend` or CLI `--backend` to select a primary GGML backend
explicitly. `auto` (the default) preserves the policy below; `cpu`, `opencl`,
and `hexagon` override `n_gpu_layers`. `hexagon` selects the `HTP0` device from
the `HTP` registry. An exact GGML device name such as `HTP0` is also accepted.
An unavailable or failed explicit selection fails loading instead of selecting
another backend. The existing per-operation CPU scheduler still handles
unsupported operations; selecting HTP0 does not claim every operation ran there.
On Android, backend discovery also prepends `backends_dir` to FastRPC's
semicolon-separated `DSP_LIBRARY_PATH`, preserving existing user paths and
vendor defaults. This happens even when the first model requests CPU/OpenCL,
so later models can explicitly select Hexagon. If only legacy
`ADSP_LIBRARY_PATH` is set, its entries are retained. Set `backends_dir` before
the first Engine/loader call; registry discovery remains process-global.

Hexagon is experimental and currently targets Parakeet CTC 0.6B Q8_0 and
MOSS-Transcribe-Diarize. Build and ship the matching GGML Hexagon backend and
DSP library. MOSS-Transcribe-Diarize's requirements, graph layout, device tests
and measurements are in [its guide](moss-transcribe.md#hexagon-npu). For
Parakeet CTC device validation, run
`parakeet --backend hexagon --verbose --bench --bench-json result.json`, with the
existing `--backends-dir` pointing at the installed backend libraries. The JSON
records the actual backend (`ggml-htp0`), transcript and stage timings. Use a
separate `GGML_HEXAGON_PROFILE=1` run to collect DSP operation evidence for
`IM2COL`, `CONV_2D_DW`, and eligible HMX matrix operations; backend identity alone
does not establish kernel execution or correct transcription. Measure CPU and
OpenCL with `--backend cpu` and `--backend opencl` using the same model/audio.
`Engine::backend_device()` identifies Hexagon as `BackendDevice::NPU`, while
`Engine::backend_name()` reports the resolved device name (`HTP0`).

Runtime tiering is:

1. Prefer OpenCL for Adreno 700+, where it is validated ahead of Vulkan.
2. Otherwise choose the first registered non-OpenCL GPU, such as Vulkan,
   Metal, CUDA, or Mali Vulkan.
3. Use non-Adreno OpenCL only when no non-OpenCL GPU can initialize.
4. Fall back to CPU.

Adreno 6xx OpenCL is skipped because it produces incorrect output. Set
`PARAKEET_ALLOW_ADRENO_6XX=1` to opt in explicitly. Mali uses Vulkan for the
encoder and CTC/TDT/EOU computation, but the Sortformer diarization head is
routed to CPU because that head is incorrect on Mali Vulkan.

TDT predictor/joint decoding uses ggml graphs on every backend, including
ggml-cpu, where the quantised joint matmuls run about ten times faster than the
former host f32 gemv loop. ggml-opencl drops the in-place state writes those
graphs rely on, so with the encoder on OpenCL the TDT and RNN-T decoder graphs
run on the CPU backend from a CPU-resident copy of the decoder weights that
the loader makes (`model_tdt_decoder_on_cpu`; the fit projector counts the
copy); Nemotron keeps the host scalar decoder there, and
`PARAKEET_TDT_HOST_DECODE=1` forces the host decoder elsewhere for parity
testing. EOU decoding uses graphs on Metal, Vulkan and CUDA and the scalar
path on CPU and OpenCL.

Nemotron 3.5 ASR streaming is supported on OpenCL (Adreno 700+). The encoder
runs on the GPU and the cache-aware streaming operating points (80, 160, 320,
560, 1120 ms) are honoured. The transducer decode runs host-side on OpenCL,
same as TDT and EOU, because ggml-opencl drops the in-place `ggml_cpy` writes
that carry the persistent LSTM state. Adreno 6xx remains blocked by default;
opt in with `PARAKEET_ALLOW_ADRENO_6XX=1` (unvalidated).

Nemotron 3.5 ASR streaming is supported on CUDA. The encoder and the
transducer decode both run on the GPU at all five operating points, with the
fused LSTM-cell and transducer-step decode running up to eight greedy steps
per graph. Validated against the NeMo stream-step references at
80/160/320/560/1120 ms on an RTX 3090 (`test-nemotron-stream-step-cuda*`).

Nemotron 3.5 ASR streaming is also supported on Vulkan. The encoder and the
transducer decode both run on the GPU at all five operating points, and the
decode takes the same fused LSTM-cell and transducer-step graphs as CUDA and
Metal because ggml-vulkan implements `GGML_OP_LSTM_CELL` and
`GGML_OP_TDT_STEP`. Validated against the NeMo stream-step references at
80/160/320/560/1120 ms on an RTX 3090 (`test-nemotron-stream-step-vulkan*`).

Nemotron 3 Diarization runs on CUDA, Vulkan, and Metal. The feature-stacking
projection, the 31-layer RoPE encoder with fused flash attention, the subpixel
upsampler, and the speaker head form one GPU split, and offline inference and
cached streaming pass the NeMo-Speech.cpp reference test
(`test-nemotron-diarization-gpu`) on an RTX 5090 (CUDA and Vulkan), an AMD
Radeon RX 7600 XT (Vulkan), and an Apple M3 Ultra (Metal). When the active
backend cannot run fused flash attention for this graph, the encoder uses an
unfused attention (matmul, softmax, matmul) that stays on the GPU instead of
sending 31 attention ops per call to the CPU. ggml-opencl takes that path on
Adreno unless `GGML_OPENCL_FA_ADRENO=1` is set. Adreno's ggml-opencl and
ggml-vulkan matmul kernels narrow activations to half precision by default,
which put the Adreno 830 at the edge of the reference tolerance (relative L2
0.0030 on OpenCL, 0.0040 on Vulkan against 0.003), so on Adreno the graph
requests `GGML_PREC_F32` on every matmul; that halves the error for a 2-3 %
cost. Both Adreno backends then pass the reference test on a Snapdragon 8
Elite, and OpenCL, which the tier policy selects on Adreno 700+, runs the
model about twice as fast as its CPU. ggml-vulkan on that GPU is slower than
the CPU for this model. Mali Vulkan is untested on hardware; the Mali routing
of the Sortformer head does not apply to this model. See
[Nemotron 3 Diarization on GPU](performance.md#nemotron-3-diarization-on-gpu)
for timings and accuracy.

The graph decoder adapts to what the active backend reports through
`ggml_backend_supports_op`, probed once at load. Where the backend runs the
fused LSTM cell (`GGML_OP_LSTM_CELL`) and the transducer step control
(`GGML_OP_TDT_STEP`), which ggml-speech implements for CPU, CUDA, Metal and
Vulkan, the TDT decoder runs up to eight greedy steps per graph: the joint,
argmax, LSTM update, duration bookkeeping and state selection stay on the
device and the
host reads the emitted tokens back once per graph. A backend without those
ops keeps one graph per step. Either path produces the same token sequence as
the sequential loop; the `test-tdt-unroll-parity` and `test-tdt-lstm-parity`
harnesses guard that. The encoder applies the same rule to the conformer's
depthwise convolution (`GGML_OP_CONV_2D_DW` in place where the backend
reports it, `im2col` and matmul elsewhere; the subsampler switches on a GPU
that passed the probe and on ggml-cpu, Mali and OpenCL keep the im2col
lowering) and to the gated
GLU (`a * sigmoid(b)` as one op where the backend reports it). Fused attention
is a build option (`PARAKEET_FLASH_ATTN`) that CUDA, Metal and Vulkan take, and
only after the backend accepts the exact node the encoder builds; CPU and
OpenCL keep the unfused graph in every build. A cached encoder graph computes
the 24 per-layer positional projections once per graph size (up to 256 MiB of
projections per graph) and the attention blocks read them from that buffer on
every forward; `PARAKEET_POS_PROJ_CACHE=0` recomputes them in-graph instead,
and `test-pos-proj-cache-parity` checks the encoder output is byte-identical
either way. The mel front-end runs on up
to eight host threads with output byte-equal to the single-thread result.

The CUDA path was validated on an RTX 3080 (TDT q8_0 and q4_0 transcripts,
Sortformer and streaming output byte-equal to the pre-change build, LibriSpeech
WER within noise of the CPU reference) but hardware decoder parity is not yet
covered by CI.

`GGML_BACKEND_DL=ON` builds load backend modules at runtime. Android defaults
this on and builds Vulkan, OpenCL, and CPU variants. Pass a module directory via
CLI `--backends-dir` or `EngineOptions::backends_dir`. Backend registration is
process-global: the first `Engine` construction loads from that directory and
later engines reuse the populated registry. Leave it empty for ggml's default
search path. In static `GGML_BACKEND_DL=OFF` builds the setting is a no-op.

`Engine::backend_name()` reports the ggml backend used by the decoder.
`Engine::encoder_backend()` reports the loaded encoder-sidecar configuration;
benchmark JSON additionally reports whether every measured invocation actually
completed through Core ML.

Example backend configurations:

```bash
cmake -S engines/parakeet -B build-metal -DGGML_METAL=ON
cmake -S engines/parakeet -B build-cuda -DGGML_CUDA=ON
cmake -S engines/parakeet -B build-vulkan -DGGML_VULKAN=ON
cmake -S engines/parakeet -B build-opencl -DGGML_OPENCL=ON
```

## Core ML encoder sidecar

`PARAKEET_COREML=ON` is Apple-only. It enables optional CTC/IndicConformer,
Unified RNN-T, TDT, EOU, Nemotron, and tagged Sortformer v2.1 FastConformer
encoder sidecars, and the MOSS-Transcribe-Diarize audio encoder sidecar, whose
export and routing are in [its guide](moss-transcribe.md#core-ml-encoder-sidecar).
Mel preprocessing, the CTC/RNN-T/EOU/Nemotron decoders, and the Sortformer
transformer/speaker head remain in the normal ggml pipeline; the TDT decoder
can additionally run on the Neural Engine through the
[batched decoder sidecar](#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars).
Sortformer v1 and v2 have no sidecar. The English CTC checkpoints satisfy the
CTC/IndicConformer export contract; among CTC checkpoints, the Core ML parity
tests cover only IndicConformer.

Create an export environment with versions supported by Core ML Tools. NumPy 2
is not currently compatible with its TorchScript scalar conversion, and the
optional grouped-channel LUT pass uses scikit-learn:

```bash
python3.11 -m venv .venv-coreml
. .venv-coreml/bin/activate
python -m pip install --upgrade pip
python -m pip install -r engines/parakeet/scripts/requirements-coreml.txt
```

The compiled sidecar must sit next to the GGUF and use this name:

```text
<model-basename-with-quant-stripped>-encoder.mlmodelc
```

For example, both `parakeet-tdt-0.6b-v3.f16.gguf` and
`parakeet-tdt-0.6b-v3.q8_0.gguf` resolve to
`parakeet-tdt-0.6b-v3-encoder.mlmodelc`.
Sortformer v2.1 may also load
`diar_streaming_sortformer_4spk-v2.1-encoder.mlmodelc` for exact-shape batch
inference and
`diar_streaming_sortformer_4spk-v2.1-encoder-bypass-pre-encode.mlmodelc` for
masked AOSC inference.

Export and compile a fixed-shape sidecar:

```bash
python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/parakeet-tdt-0.6b-v3.f16.gguf \
  --n-mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/parakeet-tdt-0.6b-v3-encoder.mlpackage \
  --compile-dir engines/parakeet/models
```

IndicConformer uses the same fixed-capacity contract. Core ML runs the encoder;
the multilingual CTC projection and `--language` token mask remain on ggml:

```bash
python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/indic-conformer-600m-multilingual.f16.gguf \
  --n-mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/indic-conformer-600m-multilingual-encoder.mlpackage \
  --compile-dir engines/parakeet/models
```

Nemotron supports an exact-shape offline encoder. Export it using `--wav` or
`--n-mel-frames`. Longer offline inputs and native cache-aware streaming run on
the cache-aware ggml path.

Unified RNN-T uses the same fixed-capacity contract:

```bash
python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/parakeet-unified-en-0.6b.q8_0.gguf \
  --n-mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/parakeet-unified-en-0.6b-encoder.mlpackage \
  --compile-dir engines/parakeet/models
```

For EOU, first download and convert the checkpoint, verify the F16 round trip,
then compile the exact-shape sidecar from that GGUF:

```bash
engines/parakeet/scripts/download-all-models.sh eou

python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/parakeet_realtime_eou_120m-v1.nemo \
  --hf-repo nvidia/parakeet_realtime_eou_120m-v1 \
  --out engines/parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf \
  --quant f16

python engines/parakeet/scripts/verify-gguf-roundtrip.py \
  --nemo engines/parakeet/models/parakeet_realtime_eou_120m-v1.nemo \
  --gguf engines/parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf

python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/parakeet_realtime_eou_120m-v1.nemo \
  --hf-repo nvidia/parakeet_realtime_eou_120m-v1 \
  --out engines/parakeet/models/parakeet_realtime_eou_120m-v1.q8_0.gguf \
  --quant q8_0

python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf \
  --wav engines/parakeet/test/samples/jfk.wav \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/parakeet_realtime_eou_120m-v1-encoder.mlpackage \
  --compile-dir engines/parakeet/models
```
For Sortformer v2.1, export the batch and AOSC block-stack sidecars separately:

```bash
python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/diar_streaming_sortformer_4spk-v2.1.f16.gguf \
  --wav engines/parakeet/test/samples/diarization-sample-16k.wav \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/diar_streaming_sortformer_4spk-v2.1-encoder.mlpackage \
  --compile-dir engines/parakeet/models

python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/diar_streaming_sortformer_4spk-v2.1.f16.gguf \
  --bypass-pre-encode --n-encoder-frames 410 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/diar_streaming_sortformer_4spk-v2.1-encoder-bypass-pre-encode.mlpackage \
  --compile-dir engines/parakeet/models
```


Benchmark fixed lengths and inspect ANE/GPU/CPU placement:

```bash
python engines/parakeet/scripts/bench-encoder-coreml.py \
  --gguf engines/parakeet/models/parakeet-tdt-0.6b-v3.f16.gguf \
  --mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16

python engines/parakeet/scripts/bench-encoder-coreml.py \
  --gguf engines/parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf \
  --mel-frames 1101 \
  --palettize-bits 6 --palettize-group-size 16
```

The default export uses Float16 input, output, weights, and intermediates.
CTC/IndicConformer, Unified RNN-T, and TDT shorter inputs are zero-padded to the
exported mel-frame capacity, while longer offline inputs are automatically
divided into overlapping windows that fit that capacity. EOU uses exact-shape
routing because padding future frames can change token and end-of-turn
decisions: only calls at the compiled mel-frame count use Core ML. Shorter
inputs, longer offline inputs, and mismatching streaming windows use ggml: EOU
never pads, and windowed causal/chunked attention parity has not been validated
for oversized inputs. The TDT example uses this addon's 15-second shape (1501
mel frames; its centred-STFT frontend emits `1 + samples/hop`) and optional
6-bit grouped-channel LUT weights. Grouped palettization requires coremltools 8+
and macOS 15 / iOS 18; omit both `--palettize-*` arguments for a macOS 13 / iOS
16 compatible Float16 model. `--flexible` exports a CTC/IndicConformer, Unified
RNN-T, or TDT RangeDim model, but it is a correctness/experimentation path:
measured flexible graphs place no operations on ANE and can be substantially
slower than ggml Metal. Flexible EOU export is rejected.

Nemotron routing requires the exact exported mel-frame count because its
causal, chunk-limited attention geometry is baked into the compiled program.
Longer offline inputs use the native cache-aware ggml encoder instead of
windowing the sidecar, because an independent exact-shape window cannot
preserve the encoder's full-depth receptive field. Shorter inputs are never
padded, and native cache-aware streaming stays on ggml. Flexible Nemotron
exports are rejected.

Sortformer batch routing requires the exact exported mel-frame count because
its graph has no validity-mask input; shorter and mismatched batch inputs stay
on ggml. The AOSC bypass sidecar masks padded keys and accepts encoder slabs up
to its exported capacity (410 frames for the default cache/FIFO/chunk geometry);
larger or custom geometries stay on ggml. Flexible Sortformer exports are
rejected.

At runtime the sidecar lets Core ML use all compute units and reuses its input,
feature-provider, and fixed-shape output-backing objects across predictions. The
graph is predominantly Neural Engine-backed, but a small number of operations
may prefer the GPU; forcing CPU + Neural Engine can move those operations onto
the CPU and reduce the speedup. Set `PARAKEET_COREML_COMPUTE_UNITS=cpu_and_gpu`,
`cpu_only`, or `cpu_and_ane` to override the default for placement comparisons.
The [Neural Engine exports](#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars)
are the exception: they are fully ANE-shaped and load pinned to CPU + Neural
Engine, since `all` is only a preference that may place them on the GPU;
`PARAKEET_COREML_COMPUTE_UNITS=all` restores Core ML's own choice.

A missing sidecar, load failure, incompatible shape, or runtime prediction
failure falls back to the ggml encoder. A failed AOSC bypass prediction also
quarantines that sidecar for the lifetime of the engine, so later chunks go
directly to ggml. Set `PARAKEET_COREML_DISABLE=1` to
force ggml, including for parity or benchmarking. Setting
`EngineOptions::long_form_window_frames` below zero disables automatic
windowing; an input larger than a fixed Core ML sidecar then falls back to the
single-pass ggml encoder.

For an unambiguous CTC/IndicConformer, Unified RNN-T, TDT, EOU, or Nemotron
benchmark, configure the exact build directory with Core ML enabled, compile
the sidecar beside the GGUF, and require Core ML. `--require-coreml` rejects
Sortformer GGUFs; their Core ML parity is covered by the diarization tests
instead:

```bash
cmake -S engines/parakeet -B build-parakeet-coreml \
  -DCMAKE_BUILD_TYPE=Release \
  -DPARAKEET_COREML=ON \
  -DGGML_METAL=ON
cmake --build build-parakeet-coreml --target parakeet-cli -j

./build-parakeet-coreml/parakeet \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --wav engines/parakeet/test/samples/jfk.wav \
  --n-gpu-layers 999 \
  --bench --bench-warmup 2 --bench-runs 5 \
  --bench-json /tmp/parakeet-tdt-coreml.json \
  --require-coreml --verbose
```

The JSON may still contain `"backend": "ggml-metal"` because the decoder
continues to use Metal. Confirm encoder execution using `encoder_backend` and
`encoder_coreml_all_runs`. A `coreml-all` encoder label means Core ML may place
operations across ANE, GPU, and CPU; it does not mean ANE-only execution.

The desktop macOS benchmark generates this sidecar from the downloaded F16 GGUF
on its first run and stores the compiled bundle in the GitHub Actions cache.
Later runs restore that cache and go directly to the benchmark comparison.

Windowing bounds Core ML input shapes and memory, but full-context attention is
then local to each overlapping window. The stitched result should therefore be
treated as close to, rather than bit-identical with, a single full-length encode;
validate accuracy on representative long recordings before production use.

## Neural Engine fixed-window encoder and TDT decoder sidecars

TDT models can run on the Apple Neural Engine end to end with two sidecars that
TranscribePsy publishes ready-made
([qvac/TranscribePsy-250M-EU-sub-coreml](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub-coreml);
`scripts/download-all-models.sh transcribepsy` fetches them) and that
`export-encoder-coreml-ane.py` / `export-decoder-coreml-ane.py` export from any
offline TDT GGUF. Both sit next to the GGUF under the usual derived names:

```text
<model>.gguf
<model>-encoder.mlmodelc
<model>-decoder.mlmodelc
```

The **fixed-window encoder** is a channel-first `[1, C, 1, T]` graph over a
fixed window (`--window-mel-frames`, 1500 mel frames = 15 s by default) with
runtime masks: `mel_mask` zeroes the padded mel frames and, through its
even-index slices, the padded frames at every subsampling resolution;
`pad_mask` zeroes the padded encoder frames and `key_bias` masks them out of
attention, so a partially filled window uses the same graph and its valid
frames come out as the reference encoder's for the short input alone (the
exporter's `--check` verifies both a full and a partial window). The wrapper
derives all three masks from the frames that hold audio, so the mel
front-end's zeroed trailing padding frame counts as padding as well. The published
TranscribePsy export predates `mel_mask` and masks after subsampling only; the
wrapper fills `mel_mask` when an export declares it. The wrapper
recognises the inputs (`mel`, `pad_mask`, `key_bias`, optionally `mel_mask`), reports the window
as the sidecar's fixed capacity, and the engine's long-form plan tiles longer
audio across overlapping windows with 16 encoder frames (1.28 s) of seam
context each side, the setting the export was validated with
(`PARAKEET_COREML_ANE_CONTEXT_FRAMES` overrides it; an explicit
`EngineOptions::long_form_context_frames` wins). Weights are 6-bit palettized
(`--nbits 0` keeps fp16). Unlike the single-input export, this layout is fully
Neural Engine-resident on the published TranscribePsy model. The short window
and seam context cost accuracy on long recordings without pauses; the
[measurements](performance.md#transcribepsy-on-the-apple-neural-engine-mac-mini-m4-2026-10)
quantify it against the CPU windows.

```bash
python engines/parakeet/scripts/export-encoder-coreml-ane.py \
  --gguf engines/parakeet/models/transcribepsy-250m-eu-sub.gguf \
  --out  engines/parakeet/models/transcribepsy-250m-eu-sub-encoder.mlpackage \
  --compile-dir engines/parakeet/models --check
```

The **TDT decoder sidecar** runs the prediction LSTM and the joint for 16
independent decoder states (`--lanes`) per dispatch; the prediction network is
memory-bandwidth bound, so one weight read serves every lane. The TDT walk and
the token-embedding lookup stay on the host (a gather is not a Neural Engine
op). The published model keeps the joint output projection at 6 bits too
(`--nbits-out 6`); the exporter's default leaves it fp16.

```bash
python engines/parakeet/scripts/export-decoder-coreml-ane.py \
  --gguf engines/parakeet/models/transcribepsy-250m-eu-sub.gguf \
  --out  engines/parakeet/models/transcribepsy-250m-eu-sub-decoder.mlpackage \
  --compile-dir engines/parakeet/models --nbits-out 6 --check
```

With the decoder sidecar loaded, `Engine::transcribe*` on a TDT model takes the
**independent-segment path**: the mel is cut into segments no longer than the
encoder window, each cut at the quietest mel frame within the last 6 s before
the cap so the state reset falls in a pause rather than mid-word; every segment
is encoded in one dispatch and decoded from a fresh prediction state, 16
segments per decoder dispatch; a segment that emits nothing is re-decoded
warm-started from the previous segment's state; the texts are joined. The
output can differ from the sequential decoder's. `EngineResult::segment_plan_ms`
and `assemble_ms` report the planning and stitching time. The CLI's one-shot
path keeps the sequential decoder over the windowed encoder output.
Overrides: `PARAKEET_ANE_BATCH_DECODE=0` keeps the sequential path,
`PARAKEET_ANE_SEGMENT_MS` and `PARAKEET_ANE_PAUSE_SEARCH_MS` change the cap and
the pause search, and `PARAKEET_ANE_DECODE_HOST=1` decodes the same segments on
the host for parity checks. The lane walk's geometry is cross-checked against
the GGUF at load; a mismatched export is refused with a log line and the
sequential decoder runs.

A **GGUF without encoder weights**, such as the 17 MB TranscribePsy Core ML
GGUF, loads only when the encoder sidecar can take the encoder over: it is then
loaded before weight allocation and the `encoder.*` tensors get no memory.
Without the sidecar, or on a build without `PARAKEET_COREML=ON`, loading fails
with a message saying so, and every path that would reach the ggml encoder
(`--profile`, the memory-fit projection) fails instead of running. The same
sidecars work next to a full GGUF renamed to match them; set
`PARAKEET_COREML_ONLY=1` so the ggml copy of the encoder weights, which the
Neural Engine path never reads, is not loaded.

## Core ML family routing

| Family | Contract | Inputs the sidecar does not take |
|---|---|---|
| CTC (English, IndicConformer), Unified RNN-T, TDT 0.6B-v3 / 1.1B | fixed capacity: shorter inputs are zero-padded to the compiled shape, longer offline inputs are split into overlapping windows | Unified RNN-T cache-aware streaming runs on ggml |
| TDT with a fixed-window Neural Engine encoder (TranscribePsy) | fixed window with padding masks: shorter inputs are padded and masked, longer offline inputs are split into overlapping windows with 16 frames of seam context, or into pause-cut independent segments when the TDT decoder sidecar is present | streaming runs on ggml; the decoder sidecar serves only the engine's offline transcribe path |
| EOU | exact compiled shape only | every other length, including mismatched streaming windows, runs on ggml; EOU never pads |
| Nemotron | exact compiled shape only | longer offline inputs and streaming take the cache-aware ggml path |
| Sortformer v2.1 (`sortformer-streaming-v2.1-aosc`) | exact-shape batch sidecar plus a masked AOSC sidecar | batch inputs of any other length, and AOSC slabs above the masked capacity, run on ggml |
| Sortformer v1, v2 | no sidecar | always ggml |
| MOSS-Transcribe-Diarize | Whisper encoder plus adaptor per 30 s window, the same zero-padded window the ggml encoder runs; separate `MOSS_COREML_*` switches | none; the Qwen3 decoder always runs on ggml. See [MOSS-Transcribe-Diarize](moss-transcribe.md#core-ml-encoder-sidecar) |

## IndicConformer and Unified Core ML export

IndicConformer CTC uses the fixed-capacity sidecar contract shared by the
offline TDT and Unified encoders. The FastConformer stack runs in Core ML,
while the multilingual CTC vocabulary projection and language mask stay on
ggml. Short inputs are padded to the compiled capacity and long inputs use the
overlapping encoder-window plan:

```bash
python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/indic-conformer-600m-multilingual.f16.gguf \
  --n-mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/indic-conformer-600m-multilingual-encoder.mlpackage \
  --compile-dir engines/parakeet/models
```

Place the compiled directory beside any quantization as
`indic-conformer-600m-multilingual-encoder.mlmodelc`. Missing or incompatible
sidecars and prediction failures fall back to the complete ggml path.

Unified RNN-T uses the same fixed-capacity sidecar contract as TDT. Shorter
inputs are zero-padded to the exported capacity, and oversized inputs use the
existing overlapping long-form window plan:

```bash
python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/parakeet-unified-en-0.6b.q8_0.gguf \
  --n-mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/parakeet-unified-en-0.6b-encoder.mlpackage \
  --compile-dir engines/parakeet/models
```

Place the compiled directory beside the GGUF as
`parakeet-unified-en-0.6b-encoder.mlmodelc`. Missing or incompatible sidecars
and prediction failures fall back to ggml.
