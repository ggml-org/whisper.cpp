# parakeet engine: command-line tools

Part of the [parakeet engine documentation](../README.md).

## CLI and microphone examples

The file CLI accepts:

```text
parakeet --model <model.gguf> (--wav <16-kHz-mono.wav> |
         --pcm-in <raw> --pcm-format s16le|f32le --pcm-rate 16000) [options]
```

Useful groups include `--threads`, `--n-gpu-layers`, `--backend`, `--backends-dir`,
`--language`, `--stream`, `--stream-duplex`, context/chunk options,
`--diarization-model`, OpenCL environment controls, `--bench`, `--profile`,
and `--dump-mel`. Run `parakeet --help` for the complete list.
`--require-coreml` (with `--bench`, CTC/IndicConformer, Unified RNN-T, TDT,
EOU, and Nemotron only) fails unless every timed encoder invocation ran on the
[Core ML sidecar](backends.md#core-ml-encoder-sidecar). `--bench`
covers the transcription models only: the diarization path (a Sortformer GGUF
at `--model`) and the attributed path (`--diarization-model`) return before
the bench loop, ignoring the `--bench*` flags — time the invocation externally
to benchmark those.

### Long-form window flags

Offline transcription encodes an input longer than one window in overlapping
windows, so the encoder's memory depends on the window rather than on the
length of the recording. `--window-sec S` sets the window (at least 256 encoder
frames, 20.5 s at the 80 ms stride; at most the model's positional range, 400 s
for TranscribePsy) and `--overlap F` the fraction of each window shared with its
neighbours, from 0 to 0.5. Without the flags the model's own recommended
window applies when its GGUF carries one, which TranscribePsy's does (60 s
with 25 % overlap, the setting the research team validated on phones);
otherwise the engine's automatic window applies (the fewest windows of at most
300 s, or the model's range if shorter, balanced to equal length, with 256
frames of context each side). `--window-sec 0` encodes the whole input in one pass, whose memory grows
with the square of its length. Inputs shorter than the window are encoded in
one pass at their own length either way. `--stream`, `--diarization-model` and
`--profile` run their own encoder passes and ignore both flags with a warning.

`--window-sec 60 --overlap 0.25` is the setting TranscribePsy was validated
with on phones. On a 412 s recording transcribed with TranscribePsy q8_0 on an
x86-64 CPU with eight threads, it cut the encoder time from 18.0 s to 10.8 s
and peak memory from 2.4 GB to 0.9 GB against the automatic 300 s window:

```bash
build-parakeet/parakeet \
  --model engines/parakeet/models/transcribepsy-250m-eu-sub.gguf \
  --wav long.wav --threads 6 --window-sec 60 --overlap 0.25
```

`--backend hexagon` explicitly requests HTP0, even when OpenCL is present.
`--backend cpu` and `--backend opencl` select comparison backends. Explicit
selection overrides `--n-gpu-layers` and fails if unavailable; omit the option
or use `--backend auto` to preserve automatic selection. See [backends](backends.md)
for the distinction between selected backend and measured DSP operation execution.

```bash
build-parakeet/parakeet \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --wav engines/parakeet/test/samples/jfk.wav --n-gpu-layers 1

build-parakeet/parakeet \
  --model engines/parakeet/models/parakeet_realtime_eou_120m-v1.q8_0.gguf \
  --wav engines/parakeet/test/samples/jfk.wav \
  --stream --stream-duplex --emit jsonl

build-parakeet/parakeet \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --diarization-model engines/parakeet/models/diar_sortformer_4spk-v1.f16.gguf \
  --wav engines/parakeet/test/samples/diarization-sample-16k.wav

# standalone Sortformer diarization: speaker segments only, no ASR model
build-parakeet/parakeet \
  --model engines/parakeet/models/diar_sortformer_4spk-v1.f16.gguf \
  --wav engines/parakeet/test/samples/diarization-sample-16k.wav

# IndicConformer: --language is required and selects the token range
build-parakeet/parakeet \
  --model engines/parakeet/models/indic-conformer-600m-multilingual.q8_0.gguf \
  --wav engines/parakeet/test/samples/hi-16k.wav --language hi

# Nemotron: locale alias or auto; empty also selects auto
build-parakeet/parakeet \
  --model engines/parakeet/models/nemotron-3.5-asr-streaming-0.6b.f16.gguf \
  --wav engines/parakeet/test/samples/jfk.wav --language en-US
```

`live-mic` runs CTC/RNN-T/TDT/EOU/Nemotron transcription or Sortformer diarization.
`live-mic-attributed` combines a CTC/RNN-T/TDT/EOU ASR model with Sortformer.
Both capture 16 kHz mono through miniaudio, accept independent streaming and
backend options, and finalize tail audio on Ctrl-C.

## Memory-fit preflight (`parakeet-fit-params`)

`parakeet-fit-params` projects whether a model fits the device memory
available right now, **without loading any weights**: it reads only GGUF
metadata, drives the same loader/graph builders as a real run through ggml's
size-only allocator APIs, and compares the projection against the device's
free memory. Library API in `<parakeet/fit.h>` (`parakeet::fit_params`);
status semantics follow the SDK's `@qvac/model-fit` contract
(0 = fits, 1 = does not fit, 2 = error — also the exit code).

```bash
# CPU projection for the default 300 s workload
build-parakeet/parakeet-fit-params \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf

# GPU projection, machine-readable
build-parakeet/parakeet-fit-params \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --n-gpu-layers 1 --json

# Explicit Snapdragon Hexagon projection
build-parakeet/parakeet-fit-params \
  --model engines/parakeet/models/parakeet-ctc-0.6b.q8_0.gguf --backend hexagon
```

`--backend` (`FitOptions::backend`) takes the same names as the engine's
`--backend`: `auto` follows `--n-gpu-layers`; `cpu`, `cuda`, `vulkan`, `opencl`,
`webgpu`, `hexagon` or an exact ggml device name projects for that backend, and
an unavailable one is an
error with reason `no-backend-device`, as a real load fails. A Hexagon session
allocates host memory and reports none of its own, so the verdict reads the
CPU device's free memory and charges host buffers to the same pool.

The workload (`--audio-seconds`, default 300) sets the longest single
transcribe input to project for. Device-side memory is bounded by the
long-form encoder window, so the device projection saturates once the audio
exceeds one window; host-side buffers (full-input mel, stitched encoder
output) keep growing with it. The projection conservatively sums up to three
window graph allocations. Windowed inference limits the encoder graph cache
to one graph during the pass, so this part of the projection can overestimate
the runtime's peak allocation.
Sortformer diarization does not window — its device projection grows with
the full input (`O(T^2)` head attention).

Nemotron is fully modelled, and both of its exceptional properties are
priced: its device projection also grows with the audio (the locale-prompt
projection graph runs over the **full** stitched encoder output, not per
window), and its native cache-aware streaming is projected explicitly — the
result covers one offline transcribe **plus** one live streaming session
(step graph, per-chunk pre-encode graph, per-layer channel/time caches and
the other host-resident session state). `--nemotron-chunk-ms` selects the
streaming operating point to project (one of the GGUF's allowed values,
80/160/320/560/1120 on the shipped checkpoint); the default projects the
largest operating point, which bounds every other.

Nemotron 3 Diarization is modelled as well. Offline inputs up to the
long-form window (90 s by default, `--window-frames` overrides it) run as one
graph, so the device projection grows with the audio up to the window and
saturates beyond it, where the engine switches to its long-form path. Every
graph shares the model scheduler, so the compute figure is the largest of
the single-pass graph, one chunk at the default live geometry (1040 ms
chunk, 80 ms right context, 264 speaker-cache and 80 FIFO rows), and, for
longer workloads, one 30 s long-form chunk. With long-form disabled
(`--window-frames -1`), audio past the GGUF's
`sortformer.encoder.pos_emb_max_len` encoder frames (400 s on the official
checkpoint) is an error with reason `workload-too-large`, as it is at
runtime. On a backend without fused flash attention the graph uses
unfused attention, whose score matrix grows with the square of the input
length, and the projection prices that graph.

On Apple builds, Nemotron 3 Diarization and Sortformer v2.1 also charge the
Core ML diarizer sidecars beside the GGUF that the workload can load
(`coremlBytes`, and a `core ml sidecars` line in the report). On unified
memory the verdict counts this charge;
[backends.md](backends.md#core-ml-diarizer-sidecars) gives the estimate.

The projection prices each graph as built. ggml-vulkan reorders graph nodes
inside the scheduler, which can shorten tensor lifetimes: on an Adreno 830
the Nemotron 3 Diarization streaming graph reserves 2.2 MiB (24 %) less than
projected. `test-fit-params` therefore sets `GGML_VK_DISABLE_GRAPH_OPTIMIZE`
before it compares the projection with real reservations.

The projection is exact where it can be: `test-fit-params` asserts the
projected weight, encoder-compute, Sortformer-head, Nemotron
prompt/step/pre-encode, and Nemotron 3 Diarization graph bytes equal what a
real load/encode/diarize/stream allocates, byte for byte. On Hexagon each
buffer also carries a 4 KiB guard page that ggml does not report, so the
projection is short by one page per buffer. For the legacy families (CTC/RNN-T/TDT/EOU) the
projection covers the offline paths; their streaming sessions build smaller
per-chunk graphs but rotate through the same graph cache with
session-dependent keys, so treat the offline projection as a guide, not a
proven bound, for streaming. Nemotron streaming is a modelled part of the
projection, not a guide.
