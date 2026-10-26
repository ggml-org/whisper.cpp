# tts engine: MOSS

Part of the [tts engine documentation](../README.md).

## GPU validation (CUDA and Vulkan)

All MOSS engines use the shared ggml GPU selector and GPU/CPU scheduler.
With a CUDA- or Vulkan-enabled ggml install, `--gpu` can run MOSS-TTS/TTSD,
MOSS-SoundEffect and MOSS-Speech on that backend; the selector prefers CUDA
when both are present. For a single-backend development build, build ggml
with `GGML_CUDA=ON` or `GGML_VULKAN=ON` (and the other GPU backends off),
install it, and configure the speech engine with that installation on
`CMAKE_PREFIX_PATH`. On a build with several GPU backends,
`TTS_CPP_GPU_BACKEND=cuda` or `TTS_CPP_GPU_BACKEND=vulkan` pins the shared
selector for validation:

```sh
TTS_CPP_GPU_BACKEND=cuda build/moss-cli --gpu --backbone moss-tts-delay-f16.gguf \
    --decoder moss-codec-decoder-f16.gguf --text "CUDA speech test." --out cuda.wav
ctest --test-dir build -R '^test-moss-(cuda|vulkan)$' --output-on-failure
```

`test-moss-cuda` and `test-moss-vulkan` run one binary, `test-moss-gpu`,
with the backend as its argument; each registers only when ggml carries
that backend. The test generates small random-weight GGUFs and compares CPU
and GPU outputs for Delay prefill/decode/reset, both codec halves, streaming and
reduced-channel dialogue decoding, SoundEffect text/DiT/VAE, Speech's two
LM heads and Whisper-VQ tokenizer. It covers F32, F16 and mixed Q8_0/F16
weights, plus the Speech LM's BF16 export with an F16 tokenizer. It fails
if the named backend is unavailable instead of accepting a CPU-only run or
the other GPU. The existing
manual `tts CI` workflow runs it with `run_gpu=true`; no model downloads
are required for this test.

The same workflow's `run_moss_e2e=true` lane downloads registered checkpoints
and saves generated WAVs, transcripts, logs and model hashes. It runs every
case in `moss_e2e_cases` on every backend in `moss_e2e_backends` (default
`["cuda","vulkan"]`), building ggml with only that backend so a CLI cannot
land on the other. The default cases are `tts-q8_0`, `ttsd-q8_0`,
`sfx-q8_0`, `speech-q8_0` and `transcribe-q5_0`, leaving memory headroom
on the shared 20 GiB runner. The TTS/TTSD cases download the registered F16
backbone and requantize its transformer matrices to Q8_0; embeddings,
output heads and codec weights stay at their source precision. Artifacts
include the quantizer log and hashes of both source and generated weights.
Full-precision cases remain available through `moss_e2e_cases`, but can
exhaust CUDA device memory on this runner. Cases run
serially because runner services share GPU memory. Speech cases use
`test-moss-speech-e2e` to make two replies on one engine instance, checking
that staged model unloading also works on the next request.

Each GPU invocation also takes a host-local lock keyed by the physical
GPU UUID and waits for two consecutive memory readings with at most 1 GiB
unavailable. Admission times out after 15 minutes and records the readings;
foreign processes are never terminated. This coordinates OpenMOSS runs across
runner accounts and avoids starting while another workload occupies the GPU.
Other GPU workloads do not honor this lock and can still start afterward;
before/after process snapshots remain necessary to diagnose contention. This is
best-effort admission, not exclusive tenancy or a VRAM reservation. The workflow
matrix limit covers only its own jobs. Saved runner logs show `/opt/actions-runner`,
`actions-runner-2` and `actions-runner-3` workspaces, and process snapshots include
other applications on the same 20 GiB NVIDIA GPU. Cross-team scheduling guarantees
are unknown; the gate cannot prevent a foreign allocation after admission or
prove which short-lived process caused an OOM. Timestamped before/after snapshots
and the inference error remain artifacts, and a mid-run OOM still fails the case.
No infrastructure changes or termination of foreign processes are performed.
The TTS case also saves a CPU reference with the same checkpoint, text and
seed to investigate backend-dependent generation quality.

`test-moss-vulkan-stream` exercises tiny end-to-end fixtures with 1/2/5/25
frame chunks, a 37-chunk run, final partial chunks, callback and explicit
cancellation, and reuse after each cancellation. It checks finite output,
sample counts and batch/stream waveform agreement with F16 Vulkan enabled.
The full TTS CI case runs `test-moss-tts-stream-e2e` with real checkpoints
and voice conditioning: 7/25-frame chunks, repeated requests, both cancellation
paths and a longer utterance. WAVs and callback timing CSVs are retained.
Waveform agreement establishes correct chunk delivery; listening/transcription
checks are still needed to establish that the model speaks the requested text.

The TTS case also gates every expected batch/stream/reuse WAV and its CPU
reference with the existing Whisper CPU intelligibility scorer. The reference
model is pinned by revision and SHA-256 (the benchmark's Whisper Tiny), and
the ASR executable is a separate static CPU build of the vendored Whisper/ggml
pair so its versioned CLI dependencies are isolated from the GPU test pin.
The synthesis text is never provided as an ASR prompt. Missing/unavailable
scores, non-finite scores, or word error rate above 25% fail the case. This
tolerates limited reference-ASR errors while rejecting the observed wrong-text
outputs (100% WER); it is not a naturalness or speaker-similarity metric.
Synthesis and ASR use the same prompt files. Transcripts, per-output scores
and ASR settings are saved even when a later synthesis step fails.

Unconditioned TTS has a suspected pre-existing wrong-text issue. The TTS
case also builds upstream commit `8ae24fffce8d25fe4bfcc9b8d81f51374b29e65c`
and runs the same CPU, GPU batch and GPU streaming requests with
identical checkpoints, seed and ggml on the same worker. An unconditioned
quality failure is non-blocking only when its PCM samples and audio format
exactly match that baseline. Such failures remain visible in transcripts
and `pre_existing_quality_issues`; they are not counted as quality passes.
Changed audio, a missing/unverified baseline or an ASR error cannot use this
exception. Voice-conditioned streaming and reuse retain the strict quality
gate. This keeps existing general TTS defects outside the GPU backend changes while
continuing to reject new regressions.

Lifecycle owner: Omar (`@ogad-tether`), tracked in
[wrong-text issue #328](https://github.com/tetherto/qvac-fabric-speech.cpp/issues/328).
The exception expires at **2026-11-12 00:00 UTC** and then fails closed, even for
identical baseline PCM. The result artifact includes the issue and review date.
Improved audio that meets the WER gate passes immediately without a baseline bump;
changed audio that still fails WER is a blocking regression, even if its WER is
lower than before. The baseline is never automatically advanced. Renewing the
deadline or replacing the baseline requires an intentional reviewed change linked
to #328 with old/new audio hashes, transcripts, scores and justification. When the
underlying fix passes every expected unconditioned and voice-conditioned output
on CPU/CUDA/Vulkan (including streaming and reuse), remove this exception and the
baseline workflow build/runs together, retaining the strict quality gate.

Speech cases also run `test-moss-speech-codec-e2e cuda|vulkan CODEC.gguf OUTPUT_DIR`.
This loads only the codec, comparing CPU and the selected GPU on fixed speech tokens,
identical diffusion noise and a short voice reference. It checks finite mel
values, numerical agreement, output length and signal level for both the
single-batch and batched CFG paths, and saves the audio and mel dumps.
The shared S3Gen flow decoder requests F32 flash-attention accumulation:
default F16 accumulation on MoltenVK can amplify errors across solver steps
until the mel becomes non-finite. F16 weights and other Vulkan operations
remain enabled; no `GGML_VK_DISABLE_F16` workaround is required. Non-finite
flow states are reported as synthesis errors instead of decoded as audio.

### S3Gen flash-attention precision audit

This audit applies specifically to ggml
[`e16d1332bdafcaade27eb402c62418239faef632`](https://github.com/tetherto/qvac-ext-ggml/tree/e16d1332bdafcaade27eb402c62418239faef632).
The setter changes only `op_params[3]`
([ggml.c](https://github.com/tetherto/qvac-ext-ggml/blob/e16d1332bdafcaade27eb402c62418239faef632/src/ggml.c#L6307)).
Both single and batched S3Gen CFM attention retain the explicit F32 request.

- CPU: `ggml_compute_forward_flash_attn_ext` sends DEFAULT and F32 through
  the same implementation, explicitly using F32 accumulators
  ([ops.cpp](https://github.com/tetherto/qvac-ext-ggml/blob/e16d1332bdafcaade27eb402c62418239faef632/src/ggml-cpu/ops.cpp#L10807)).
- Metal: flash-attention dispatch reads scale, max bias and softcap (slots 0–2),
  and its pipeline selection does not consume slot 3
  ([dispatch](https://github.com/tetherto/qvac-ext-ggml/blob/e16d1332bdafcaade27eb402c62418239faef632/src/ggml-metal/ggml-metal-ops.cpp#L4405),
  [pipeline](https://github.com/tetherto/qvac-ext-ggml/blob/e16d1332bdafcaade27eb402c62418239faef632/src/ggml-metal/ggml-metal-device.cpp#L1803)).
  The F32/F16 KV matrix kernels use float QK, softmax and output accumulation
  (`FA_TYPES_F32` / `FA_TYPES`), with some half intermediates. This is not a
  claim that all Metal arithmetic or BF16 paths use F32
  ([kernel types](https://github.com/tetherto/qvac-ext-ggml/blob/e16d1332bdafcaade27eb402c62418239faef632/src/ggml-metal/ggml-metal.metal#L10129)).
- CUDA: flash-attention support, allocation and kernel selection do not read the
  precision slot; the shared launch code passes only slots 0–2. DEFAULT and F32
  select the same kernels and arithmetic for the same tensors/device. This does
  not imply CUDA attention is universally F32; its existing conversions and
  architecture-specific paths remain in effect
  ([selection](https://github.com/tetherto/qvac-ext-ggml/blob/e16d1332bdafcaade27eb402c62418239faef632/src/ggml-cuda/fattn.cu#L366),
  [launch](https://github.com/tetherto/qvac-ext-ggml/blob/e16d1332bdafcaade27eb402c62418239faef632/src/ggml-cuda/fattn-common.cuh#L1191)).
- Vulkan: `f32acc` explicitly reads slot 3, so the request changes accumulation
  where the default would use F16
  ([dispatch](https://github.com/tetherto/qvac-ext-ggml/blob/e16d1332bdafcaade27eb402c62418239faef632/src/ggml-vulkan/ggml-vulkan.cpp#L13515)).

CPU/Metal/CUDA equivalence above is source-level evidence for this flag at this
pin, not measured before/after codec latency or numerical results. Existing green
GPU inference runs validate the F32-requesting code but are not A/B benchmarks.
Re-audit these dispatch paths on ggml upgrades. If a backend starts honoring the
flag differently, compare fixed-input codec-only DEFAULT/F32 builds (both CFG
paths, identical codes/noise/voice, mel/PCM finiteness/error and warmed timings)
before updating the pin; never weaken production precision to pass a comparison.
Local codec comparisons must run serially with a 3 GiB process-tree RSS ceiling
and abort on memory pressure or swap growth. Full Speech/TTS LMs are unnecessary.

These generated-model checks pass on Apple M2 through MoltenVK. They do
not establish full-checkpoint speech quality, performance, or Android and
discrete-GPU compatibility. The per-model reference tests below remain the
full-checkpoint validation path; MOSS-Speech's shared S3Gen/HiFT reply decoder
also needs the real codec checkpoint.

## MOSS Delay

[MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) is a family of
delay-pattern TTS models from OpenMOSS: a Qwen3-shaped autoregressive
backbone reads packed rows of one text token plus 32 audio codes and emits
the next row, one text head and 32 audio heads wide, and a
residual-vector-quantised codec turns the code frames into 24 kHz mono audio
at 12.5 frames per second (1920 samples per frame). The 32 audio channels are
staggered one step apart — the delay pattern — so channel `c` of frame `t`
is emitted at step `t + c`. The engine covers the `MossTTSDelay`
architecture, which is shared by MOSS-TTS-v1.5 (single speaker turn) and
MOSS-TTSD (dialogue checkpoints).

Cloning needs no speaker encoder: the codec *encoder* turns a reference wav
into code frames, and those frames are packed, delay-patterned, into the
user section of the prompt.

**Status — CPU and Metal, validated end to end against the released
checkpoints.** `test-moss-generation` pins the delay/de-delay round trip, the
in-loop drain state machine, the sampling primitives, segment extraction, and
prompt packing against the upstream semantics. Against converted
MOSS-TTS-v1.5 and MOSS-Audio-Tokenizer GGUFs, the codec encoder-to-decoder
round trip reaches 0.94 correlation on real audio, and full synthesis on
Metal produces clean speech that terminates generation naturally. The codec
transformers run banded attention: each sliding-window block attends inside
its configured context window, so attention memory scales with the window
instead of the clip length, and the banded output matches the dense
computation on real audio. Numeric reference parity and the remaining
backends are the follow-up to this change.

### Convert

Three GGUFs, because the halves have different lifetimes: the backbone and
the codec decoder are needed for every synthesis, the codec encoder only for
voice cloning. Checkpoints come from
[OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) (or a
MOSS-TTSD checkpoint with the same architecture) and
[OpenMOSS-Team/MOSS-Audio-Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer)
for the codec.

```sh
# backbone: Qwen3 layers, text embedding + 32 audio embeddings,
# text head + 32 audio heads, tokenizer vocabulary and merges
python scripts/convert-moss-delay-to-gguf.py /path/to/MOSS-TTS-v1.5 \
    --outtype f16 --outfile moss-tts-delay-f16.gguf

# codec: one file per half; each stands alone
python scripts/convert-moss-codec-to-gguf.py /path/to/MOSS-Audio-Tokenizer \
    --outtype f16 \
    --encoder-outfile moss-codec-encoder-f16.gguf \
    --decoder-outfile moss-codec-decoder-f16.gguf
```

The backbone converter flattens the checkpoint's `language_config` into the
root hyperparameters, maps `emb_ext.{i}` to `token_embd_audio.{i}` and
`lm_heads.0` / `lm_heads.k` to `output` / `output_audio.{k-1}`, and exports
the tokenizer as plain `tokenizer.ggml.tokens` / `tokenizer.ggml.merges`
arrays. The codec converter merges weight-norm parametrizations and writes
per-module metadata (patch sizes, transformer shapes, context windows) so the
engine can rebuild the module chain without hardcoded shapes.

### Run

```sh
# text only: backbone + codec decoder
build/moss-cli --backbone moss-tts-delay-f16.gguf \
    --decoder moss-codec-decoder-f16.gguf \
    --text "..." --language en --out out.wav

# streaming: chunks arrive at the terminal as they decode; the WAV is
# identical to the batch render for the same seed
build/moss-cli --backbone moss-tts-delay-f16.gguf \
    --decoder moss-codec-decoder-f16.gguf \
    --text "..." --language en --stream --out out.wav

# cloning: add the codec encoder and a reference wav (must match the
# codec sample rate; stereo is downmixed by averaging)
build/moss-cli --backbone moss-tts-delay-f16.gguf \
    --decoder moss-codec-decoder-f16.gguf \
    --encoder moss-codec-encoder-f16.gguf --ref-audio speaker.wav \
    --text "..." --language en --out out.wav

# two-speaker dialogue (MOSS-TTSD checkpoint): one reference per speaker,
# and the text carries the reference transcripts first, then the turns,
# all tagged [S1]/[S2]
build/moss-cli --backbone moss-ttsd-f16.gguf \
    --decoder moss-codec-decoder-f16.gguf \
    --encoder moss-codec-encoder-f16.gguf \
    --dialogue-ref s1.wav --dialogue-ref s2.wav \
    --text "[S1] <s1 transcript> [S2] <s2 transcript> [S1] ... [S2] ..." \
    --language en --out dialogue.wav

# on the GPU: same command plus --gpu
```

`--language` defaults to `zh`, matching the reference pipeline. The speech
is directable: `[pause 2.0s]` markers inside the text insert silences of
roughly the requested length, Pinyin (`ni3 hao3`) and IPA (`/həloʊ/`) notation
inline in the text steer pronunciation, and `--duration-tokens N` asks the
model for a target length of N codec frames (12.5 per second; 0 keeps the
length free). A target length also needs `n_vq - 1` drain steps plus the
`audio_end` and `im_end` rows, so the engine rejects a `duration_tokens`
that does not fit in `max_new_tokens` together with them. Sampling
follows the reference defaults — text temperature 1.5 / top-k 50, audio
temperature 1.7 / top-p 0.8 / top-k 25, repetition penalty 1.0 over the
audio channels, seed 1234 — and every knob is exposed as a flag
(`--text-temperature`, `--audio-top-k`, `--audio-repetition-penalty`, ...).

### Engine notes

The public API is `tts_cpp::moss::Engine` in
[`include/tts-cpp/moss/engine.h`](../include/tts-cpp/moss/engine.h):
construct with `EngineOptions` (model paths, language, sampling, `use_gpu`,
`backend` for an explicit device such as the [Hexagon NPU](#hexagon-npu),
`backends_dir` for builds that load ggml backends at runtime, `context`,
`max_new_tokens`), call `synthesize(text)` for a
`SynthesisResult` (mono PCM, sample rate, frame count, generation and decode
wall times), and `cancel()` from another thread to stop a run. One synthesis
runs at a time per instance; overlapping calls throw, and the engine rejects
a decoder whose quantizer count does not match the backbone's channel count.
`encoder_path` and `reference_audio_path` are needed only for voice cloning:
the reference WAV must already be at the codec sample rate (channels are
averaged; there is no resampling). Each request reseeds the RNG from the
options, so equal requests on one instance produce equal audio.

Dialogue runs in the reference pipeline's continuation mode: each speaker's
reference expands to its own `[Sn]` audio block in the user message, and the
assistant message opens with the concatenated references' codes truncated by
`n_vq - 1` rows, so the model continues the delay pattern seamlessly. The
references also stay in the codec as causal history: the decoder renders them
together with the generated audio and drops their samples, so the first
generated frames decode exactly as the reference pipeline's
decode-then-trim, in batch and in streaming. The codec and the delay model
validate every tensor against the declared geometry at load, and the codec
rejects a sampling rate outside 8-192 kHz; reference WAVs are capped at 60
seconds and at an absolute sample count that does not depend on the model. A backbone may use fewer channels than the codec carries
(MOSS-TTSD uses the first 16 of the tokenizer's 32 residual quantizers):
references are truncated to the backbone's channels and the decoder sums
only those levels.

`synthesize_stream(text, callback)` emits PCM chunks while generation runs:
a frame is complete `n_vq - 1` steps after its row, completed non-pad frames
accumulate, and every `stream_chunk_frames` of them decode incrementally: the
codec keeps each transformer layer's keys and values for its attention window
between chunks, so a chunk decodes only its new frames and the streamed audio
equals the batch render. Codec work stays linear in the output length. The
callback runs on the calling thread and returning false cancels the request;
`SynthesisResult::first_audio_ms` reports the latency to the first chunk and
`pcm` stays empty. The batch path uses the same incremental decode in
60-second windows once a segment exceeds that length, which bounds the
codec's graph size without changing the output.

Generation mirrors the reference loop. The prompt packs the fixed
`<user_inst>` template between `<|im_start|>`/`<|im_end|>` markers and ends
with an `audio_start` seed row; a reference expands to
`audio_start + (frames + n_vq - 1)` user-slot rows carrying the
delay-patterned codes + `audio_end`. During decoding the text channel is
masked to the control tokens valid at each step, audio channel `c` opens at
step `c`, and when the model emits the delay slot the drain state machine
forces delay slots while the channels close one per step and then forces
`audio_end`. The emitted rows are de-delayed and split into segments at
all-pad frames; each segment decodes from a fresh codec context, in batch and
in streaming alike, so one segment never attends to another.

On Metal, the device-table layout honors those text-logit requirements: it
projects only the two audio-control tokens while generating audio, omits the
text head during the forced drain, and uses the full vocabulary when needed.
Increasing control-token IDs select a strided view of the original weights;
other layouts are copied once at model load into a head with the original
weight type. Embedding tables stay on the device. Memory preflight includes
any additional packed head.

On ARM CPUs ggml accumulates f16 dot products in f16, and the backbone's
feed-forward output exceeds that range; the down projection therefore runs
on inputs scaled by a power of two and rescales its f32 result, which keeps
CPU synthesis finite and leaves GPU results unchanged.

### Core ML codec decoder

On Apple builds with `TTS_CPP_COREML=ON`, the codec decoder's transformer
stack can run as a Core ML model; the backbone, the codec encoder, and the
quantizer that sums the residual codebooks into latents stay on ggml. The
sidecar keeps every attention layer's keys and values in Core ML state, so it
needs macOS 15 or iOS 18; on older systems it is not attached.

```sh
python scripts/export-moss-codec-coreml.py \
    --gguf moss-codec-decoder-f16.gguf --compile-dir .
```

The engine looks for `<stem>.mlmodelc` beside the decoder GGUF, with any
quantization tag dropped (`moss-codec-decoder.mlmodelc`). Each call decodes 25
latent frames (2 s; `--chunk-frames` changes it) and carries a commit flag. A
full chunk commits its keys and values to the state; a shorter tail is padded
and decoded without committing, which is exact because the decoder is causal,
and the next full chunk decodes those frames again without emitting their audio
twice, so the streamed audio matches a single decode. The engine passes each
call's rotary tables and band masks, computed in fp32; the exporter only
reorders the rotary pairs. The ggml codec computes GELU with the tanh
approximation, while upstream uses erf (the two decodes agree at cosine
0.999998); the exporter follows ggml by default, and `--gelu erf` follows
upstream.

Only streams whose pieces fit in one sidecar call use it, which is
`synthesize_stream` with the default `stream_chunk_frames` of 25. The ggml codec
decodes a batch segment, or a longer streamed piece, in one pass, and that beats
the sidecar's 25-frame calls, so batch synthesis, the 60-second windows of long
segments, and streams with larger pieces stay on ggml. If a call fails, the
engine retires the sidecar, replays the stream's codes so far through the ggml
codec to rebuild its cache, and continues there without repeating audio.
`Engine::codec_on_coreml()` says whether a sidecar is attached, and
`SynthesisResult::codec_backend` reports where the request decoded: `ggml`, the
Core ML placement label, or `mixed`. The CLI appends the placement to its
summary line. `MOSS_COREML_DISABLE`, `MOSS_COREML_STRICT`, and
`MOSS_COREML_COMPUTE_UNITS` work as for the
[SoundEffect sidecars](#core-ml-dit-and-vae); the sidecar's state needs the GPU,
so under `cpu_and_ane` it does not run and the codec stays on ggml.

`bench-moss-coreml` streams 20 s of codes in 25-frame pieces through both codecs
(f16 decoder, median of five after a warm-up, measured 2026-10-08):

| Machine | ggml Metal | Core ML (default) |
|---|---:|---:|
| Mac mini M4 | 742 ms | 720 ms (1.03x) |
| M3 Ultra | 380 ms | 337 ms (1.13x) |

The audio matches the ggml codec at cosine 0.9998. The same 20 s decoded as one
batch took 512 ms on ggml and 700 ms on the sidecar on the M4 (0.73x), and
114 ms against 330 ms on the M3 Ultra (0.35x), which is why batch decodes stay
on ggml. Under `all` the M4 stream took 814 ms (0.89x).

### Hexagon NPU

MOSS-TTS-v1.5 and MOSS-TTSD run on the Snapdragon Hexagon NPU (HTP0; measured
on a Galaxy S25, Snapdragon 8 Elite, Hexagon v79) when requested explicitly:
`EngineOptions::backend = "hexagon"` or `moss-cli --backend hexagon`. There is
no fallback: construction fails when HTP0 is missing, and `--gpu` never selects
it. The backbone's transformer layers and audio heads run on the NPU, and so
does the codec (the decoder, and the encoder for cloning and dialogue); the
host embeds each row, samples it, and projects the whole text head on the few
steps that need it. It needs a `qvac-ext-ggml@speech` build with the Hexagon
backend (measured at `6bb85ad6`).

Weights go into the Hexagon repack buffer type. HTP multiplies `f32`, `f16`,
`q8_0` and `q4_0` weights; the backbone and the codec refuse other types for
the weights they place on the NPU. The text embedding table, the audio
embedding tables and the text head stay on the host in any type, mapped from
the GGUF rather than copied (they are read into memory instead when the file
cannot be mapped or a table is not 32-byte aligned): at 155,648 x 4,096
the text table and head are 1.27 GB each at `f16`, past HTP's 1 GiB buffer
limit, and HTP cannot gather rows from a quantized table. While audio is
generated the text channel only chooses between the generation and delay slot
tokens, so those two head rows run on the NPU as a small `f32` matrix; drain
steps, whose text token is forced, project no text logits, and only steps
after `audio_end` project the whole head, on the CPU. Other backends keep every
table on the device and project the whole head on every step.

Attention runs as one `FLASH_ATTN_EXT` with an F16 mask: per layer in the
backbone, whose value cache then uses the key layout, and per band in the codec.
Unfused, the score and value matmuls, their softmax and the transposed value
copy took 19 of the 44 s of a MOSS-TTSD generation over two voice references
(a 723-row prompt), and 96% of the codec decode of a 15.8 s utterance (6.4 s;
fused, 0.4 s). Replaying a 600-step
Hexagon trajectory on the CPU, the fused and unfused slot logits differ from the
CPU's by 1.42 and 1.36 on average and never reverse a slot decision.

The 8.5B backbones are 17 GB at `f16`. On a 12 GB phone, quantize the
transformer layers to `q4_0` and keep the tables and heads at `f16`:

```sh
python scripts/requantize-gguf.py moss-tts-delay-f16.gguf \
    moss-tts-delay-blk-q4_0.gguf q4_0 --name-filter blk.
```

The file is 7.0 GB (MOSS-TTSD 6.7 GB). About 4.2 GB of it sits on the NPU
besides the KV cache; the tables are paged in from the file as rows are read.
Quantizing the heads as well (no `--name-filter`, 4.8 GB) is the only variant
the CPU backend can hold beside the 1.8 GB codec on that phone, but it often
fails to stop: the 4 s sentence below ran on for 20 s on both backends.

`q4_0` transformer layers also stop less reliably than `f16` ones on every
backend. Over seeds 1-8 on the umbrella sentence below (at most 400 steps):

| Backbone | Backend | Stopped | Whisper WER 0 |
|---|---|--:|--:|
| `f16` | CPU (x86) | 8/8 | 8/8 |
| transformer `q4_0` | CPU (x86) | 7/8 | 7/8 |
| transformer `q4_0` | Hexagon | 7/8 | 7/8 |

On both backends seed 7 ran to the limit. MOSS-TTSD without references
stopped for 5 of 6 seeds on Hexagon. Bound long requests with
`max_new_tokens` or `duration_tokens`.

Speed on the Galaxy S25 (seed 1234, context 1024, 6 CPU threads, each run
started at thermal status 0; generation includes the prefill and, on Hexagon,
about 1.6 s of mapping the weights into the NPU session on the first request):

| Request | Backbone | Backend | Steps | Audio | Generate | Codec decode |
|---|---|---|--:|--:|--:|--:|
| "The quick brown fox jumps over the lazy dog near the riverbank." | transformer `q4_0` | Hexagon | 81 | 3.8 s | 8.5 s | 0.13 s |
| same | all `q4_0` | Hexagon | 81 | 3.8 s | 8.1 s | 0.12 s |
| same | all `q4_0` | CPU | 287 | 20.3 s | 38.3 s | 10.7 s |
| "Mobile speech synthesis has to balance quality, ..." (two sentences) | all `q4_0` | Hexagon | 287 | 20.3 s | 28.4 s | 0.70 s |
| same | all `q4_0` | CPU | 341 | 24.6 s | 56.6 s | 16.4 s |
| MOSS-TTSD, two voice references (723-row prompt) | transformer `q4_0` | Hexagon | 149 | 10.6 s | 17.4 s | 0.99 s |

Generation steps cost 99-101 ms on Hexagon and 133-166 ms on the CPU with the
same GGUF. A step is memory-bound on both: in steady state the NPU spends about
93 ms streaming 4.2 GB of `q4_0` weights (45-53 GB/s), so generation stays
slower than real time (one frame is 80 ms of audio). The codec decodes 20.3 s
of audio in 0.70 s on the NPU against 10.7 s on the CPU.

`test-moss-hexagon` (the `hexagon` arm of `test-moss-gpu`, registered when ggml
carries Hexagon) compares HTP0 with the CPU on generated GGUFs at `f32`, `f16`,
`q8_0` and `q4_0`: backbone prefill, decoding with slot and forced steps, both
codec halves, streaming, dialogue channels and the memory projection; it also
checks that `bf16` backbone and codec weights are refused. On the
S25 the worst relative L2 error is 0.06% for the backbone and 0.08% for the
codec. Whisper transcribed the Hexagon output of every request above, of
streaming, of single-reference cloning and of MOSS-TTSD with and without
references verbatim, apart from the runs that did not stop.

### Test

The suites build with `TTS_CPP_BUILD_TESTS` and need no model downloads.
`test-moss-generation` is pure CPU logic over the generation primitives.
`test-moss-load` builds tiny GGUF fixtures in-test and covers the accept path
plus every load-time rejection for the backbone, the codec, and the engine
pairing, including oversized architecture metadata, the duration budget, and
an f16 feed-forward sum past the half-precision range.
`test-moss-codec-stream` checks on random codec weights that incremental
decoding matches a single decode, that segments decode independently, and
that streaming decodes every frame exactly once; it also forces the fused
banded attention Hexagon uses onto the CPU and compares its batch, streamed
and encoder outputs with the unfused bands. `test-moss-hexagon-layout` forces
both [Hexagon](#hexagon-npu) layouts onto the CPU with f32, f16 and q8_0
weights: host tables (whole-head, slot and forced steps against the device
tables, also with q8_0 and q4_0 tables and head) and fused attention with the
key-layout value cache (prefill, decoding and reset against the unfused
attention); it also pins the weight types the Hexagon gate accepts. `test-moss-cancel` drives a real synthesis on the fixture models and
cancels it from another thread. `test-moss-cli` covers the flag surface and
runs the CLI end to end against the fixtures. `test-convert-moss` (Python,
registered when an interpreter is available, skips without numpy/gguf)
fabricates tiny checkpoints on disk, runs the converters, and validates the
emitted GGUFs. `test-moss-codec-coreml` covers the codec sidecar's routing
without Core ML, through a causal stand-in model: the rotary tables and band
masks, uneven streamed pieces against one causal pass, a restarted stream,
chunk-sized streams on the sidecar, batch decodes and longer pieces on ggml, a
failed chunk that retires it and resumes on ggml with the full history, strict
mode, and the sidecar name rule. `test-moss-codec-coreml-exporter` (Python; skips without
numpy or torch) decodes a tiny codec chunk by chunk through the exporter's
cached keys and values with the engine's commit protocol and checks it against
one pass and an independent NumPy decoder; with coremltools it also checks the
converted program's inputs, output, and states. The on-device check is
`test-moss-coreml-parity`, described in the
[MOSS-Speech test notes](#moss-speech).

## MOSS-SoundEffect

[MOSS-SoundEffect-v2](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
turns a text description into a sound effect of up to 30 seconds. It is a
flow-matching diffusion model: a Qwen3-1.7B text encoder embeds the prompt, a
30-layer Wan-style DiT (adaLN modulation, RoPE self-attention over the latent
frames, cross-attention to the text) predicts the velocity, and a continuous
DAC decoder turns the 128-channel latent (50 frames per second) into 48 kHz
mono audio. It is not a speech model; it shares the MOSS CLI and the Qwen
tokenizer with the Delay engine and nothing else.

**Status — CPU, Metal and the [Hexagon NPU](#hexagon-npu-generation),
validated against the reference pipeline.** With
the same noise, every stage matches the PyTorch fp32 pipeline: the text
encoder, one DiT velocity for the prompt and for the empty negative prompt,
an eight-step guided sampling loop, and the VAE decode. An f16 GGUF reaches
cosine similarity 0.99998 or better on all three backends; a q8_0 GGUF reaches
0.9995 per stage and 0.9987 after the eight sampling steps. The CPU path is
correct but slow (about ten seconds per DiT pass on an M1 Ultra, so a
100-step clip takes over half an hour); use `--gpu` where one is available.

### Convert

One GGUF holds the text encoder, the DiT, the VAE decoder, the tokenizer, and
the generation defaults.

```sh
python scripts/convert-moss-sfx-to-gguf.py /path/to/MOSS-SoundEffect-v2.0 \
    --outtype f16 --outfile moss-sfx-v2-f16.gguf
```

`--outtype` picks the type of the linear weights and of the text token
embedding: `f16` (6.4 GB) and `q8_0` (3.5 GB) are validated; `bf16` and `f32`
are also written. Norms, biases, modulation tables, snake parameters, and the
two small 1x1 projections (`dit.patch_embd`, `vae.post_quant`) stay f32, and
the VAE convolution kernels stay f16 because ggml's CPU im2col requires it.
The converter drops the text encoder's unused language-model head, folds the
VAE weight norm into plain kernels, lays out the transposed convolutions as
GEMM columns, and stores `1 / (alpha + 1e-9)` next to each snake `alpha`.

### Run

```sh
build/moss-cli --mode sfx --model moss-sfx-v2-f16.gguf \
    --text "Heavy rain falling on a tin roof with distant thunder." \
    --seconds 10 --gpu --out rain.wav
```

The defaults come from the GGUF and match the reference pipeline: 100 steps,
guidance 4.0, shift 5.0, and an empty negative prompt. `--steps`,
`--guidance`, `--shift`, `--negative-prompt`, and `--seed` (0 by default in
this mode) override them. `--seconds` must be in (0, 30] and is rounded to
0.1 s the way Python's `round` does. There is no streaming in this mode. On an M1 Ultra a
100-step clip takes about two minutes on Metal whatever its length, because
the model always denoises the full 30-second latent and crops the result.
`--backend hexagon` runs it on a Snapdragon NPU instead of `--gpu`; see
[Hexagon NPU generation](#hexagon-npu-generation).

### Engine notes

The public API is `tts_cpp::moss::SoundEffectEngine` in
[`include/tts-cpp/moss/sound_effect.h`](../include/tts-cpp/moss/sound_effect.h):
construct with `SoundEffectOptions` (model path, threads, `use_gpu`,
`backends_dir`, `backend`), call `generate(request, progress)` for a `SoundEffectResult`
(mono PCM, sample rate, text, diffusion, and decode wall times), and
`cancel()` from another thread or `false` from the progress callback to stop
a run. A request leaves `steps`, `guidance`, and `shift` at zero to use the
model defaults and rejects negative or non-finite values; guidance 1 skips
the negative branch. Prompts are capped at 8192 bytes. One generation runs at
a time per instance; overlapping calls throw.

Generation mirrors the reference pipeline. The prompt gets the training-time
suffix ` duration: X.Xs` and is cleaned the way upstream's `ftfy`-based
cleaner does it: HTML entities are unescaped (to a fixed point, or exactly
twice when the text contains `<`), curly quotes are straightened, full-width
forms and the ideographic space are narrowed, Latin ligatures are split, and
Unicode whitespace collapses to single spaces, so Chinese and typographic
prompts tokenize to the same ids as the Hugging Face tokenizer. Malformed
UTF-8 becomes U+FFFD byte by byte. Not reproduced: Unicode normalization
(NFC), mojibake repair, named entities beyond `&amp; &lt; &gt; &quot; &apos;
&nbsp;` (and named entities without a semicolon), the cp1252 mapping Python
applies to `&#128;`-`&#159;`, numeric references longer than eight digits,
halfwidth katakana and the other halfwidth forms past U+FF5E, and ftfy's
removal of the byte-order mark and control characters. The cleaned prompt is
tokenized up to 512 tokens; the text encoder
runs causally over the real tokens and the padding rows of the 512-token
context stay zero, as upstream. The empty negative prompt is an all-zero
context. The sampler is first-order Euler over the shifted flow-matching
schedule, `sigma' = shift * sigma / (1 + (shift - 1) * sigma)`, with classic
classifier-free guidance. The initial noise comes from a seeded Mersenne
Twister with a Box-Muller transform, so a seed reproduces a clip across runs
on one backend, but does not reproduce the PyTorch pipeline's clip for the
same seed. The DiT graph is built once per request and reused for every step;
the model refuses to run it once another graph has taken the scheduler. The
loader validates every tensor's shape and type against the metadata and bounds
the metadata itself (layer counts, hop length, latent length) so a malformed
GGUF fails with an error instead of aborting inside ggml. The
VAE decodes only the frames that cover the requested length, in 256-frame
windows with 32 frames of context on each side, and the output is cropped to
the exact sample count.

### Core ML DiT and VAE

On Apple builds with `TTS_CPP_COREML=ON`, the DiT and the VAE decoder can each
run as a Core ML model; the text encoder and the sampler stay on ggml. Each
stage is a separate sidecar, so either can be deployed without the other.

```sh
python scripts/export-moss-sfx-coreml.py --gguf moss-sfx-v2-f16.gguf --stage dit --compile-dir .
python scripts/export-moss-sfx-coreml.py --gguf moss-sfx-v2-f16.gguf --stage vae --compile-dir .
```

The engine looks for `<stem>-dit.mlmodelc` and `<stem>-vae.mlmodelc` beside
the GGUF, with any quantization tag dropped (`moss-sfx-v2-dit.mlmodelc`,
`moss-sfx-v2-vae.mlmodelc`). The exporter dequantizes any GGUF and writes an
fp16 program (`--precision float32` for fp32, `--palettize 8|6|4` for
palettized weights); `--parity-dir` checks the PyTorch rebuild against the
dumps of `scripts/dump-moss-sfx-reference.py` before exporting. All three MOSS
exporters use the environment in
[`engines/parakeet/scripts/requirements-coreml.txt`](../../parakeet/scripts/requirements-coreml.txt).
Conversion runs on Linux too; `--compile-dir` calls `xcrun coremlcompiler`.

The DiT sidecar computes one velocity over the full 30-second latent, the
same shape the ggml DiT always runs. The engine computes the timestep's
sinusoidal embedding in fp32 and passes it in, because fp16 cannot hold the
phase of a timestep near 1000; the exporter reorders the rotary pairs of the
query and key rows so the attention scores are unchanged. The VAE sidecar
decodes a fixed window of 320 latent frames (`--window` changes it): the ggml
decoder's 256-frame window with its 32 frames of context on each side, so each
call keeps 256 frames; windows at the ends of the latent shift inward instead
of padding. The
exporter rewrites the transposed convolutions as stride-phase convolutions,
evaluates the Snake sine with a polynomial (`--snake sin` keeps Core ML's
sine), and caps each Snake `1 / alpha` at 60000, because one released channel
has `alpha` near 8e-6 and its inverse overflows fp16.

A sidecar whose shapes do not match the model is not attached. A failed DiT
call is retired and that step and the rest of the request run on ggml; a
failed VAE window sends the whole decode to ggml. `SoundEffectEngine::dit_on_coreml()`
and `vae_on_coreml()` say whether each sidecar is attached, and
`SoundEffectResult::dit_backend` and `vae_backend` report where the request
ran: `ggml`, the Core ML placement label (`coreml-all`, `coreml-gpu`,
`coreml-ane`, `coreml-cpu`), or `mixed`. The CLI appends both to its summary
line. Three environment variables apply to every MOSS sidecar in the tts
engine: `MOSS_COREML_DISABLE` keeps all stages on ggml, `MOSS_COREML_STRICT`
makes a missing or failed sidecar an error instead of a fallback, and
`MOSS_COREML_COMPUTE_UNITS` (`all`, `cpu_and_gpu`, `cpu_and_ane`, or
`cpu_only`) picks the Core ML compute units. Unset or unrecognized, it means
`cpu_and_gpu` for every MOSS sidecar in the tts engine, unlike the Audio8 and
Supertonic sidecars, whose default is `all`: none of these stages ran faster
on the Neural Engine, and the reasons are below for each stage.

`bench-moss-coreml` against ggml on Metal (f16 GGUF, median of five after a
warm-up, measured 2026-10-08):

| Machine | Stage | ggml Metal | Core ML (default) |
|---|---|---:|---:|
| Mac mini M4 | DiT, one velocity over 30 s | 1702 ms | 1379 ms (1.23x) |
| Mac mini M4 | VAE, 30 s of audio | 2767 ms | 1624 ms (1.70x) |
| M3 Ultra | DiT, one velocity over 30 s | 324 ms | 305 ms (1.06x) |
| M3 Ultra | VAE, 30 s of audio | 607 ms | 335 ms (1.81x) |

A 100-step clip with guidance runs the DiT 200 times, so on the M4 the DiT
sidecar saves about a minute per clip. Velocities and audio match ggml at
cosine 0.999997 or better. Both stages run on the GPU even under `all`; on the
M4, `all` moved part of the VAE to the Neural Engine and made it slower, and
`cpu_and_ane` had not finished preparing the DiT after five minutes. A
128-frame VAE window, which keeps only 64 frames per call, ran at 0.98x on the
M4 and 1.05x on the M3 Ultra.

### Hexagon NPU generation

MOSS-SoundEffect-v2 runs on the Snapdragon Hexagon NPU (HTP0; measured on a
Galaxy S25, Snapdragon 8 Elite, Hexagon v79) when requested explicitly:
`SoundEffectOptions::backend = "hexagon"` or `moss-cli --mode sfx --backend
hexagon`. As for MOSS-TTS there is no fallback: construction fails when HTP0
is missing, and `--gpu` never selects it. The text encoder, the DiT and the
VAE decoder all run on the NPU. Only the 151,936-row token table stays on the
host, mapped from the GGUF (read into memory when the file cannot be mapped
or the table is not 32-byte aligned), and the host gathers the prompt's rows
from it. The loader takes `f32`, `f16`,
`bf16` and `q8_0` linear weights; HTP runs all of them but `bf16`, which is
refused at load, so both validated GGUFs run.

It needs a `qvac-ext-ggml@speech` build with the Hexagon backend at `d85713da`
or later, which carries the DSP buffer remapping fix
([tetherto/qvac-ext-ggml#124](https://github.com/tetherto/qvac-ext-ggml/pull/124));
the numbers below were measured at that commit. Without it the first DiT pass maps more 1 GiB weight
buffers than the DSP's address space or its 16 mapping slots can hold, the
DSP aborts and the request never returns; on the S25 the fault rebooted the
phone.

Attention runs as one `FLASH_ATTN_EXT`: the DiT's self-attention over the 1500
latent frames and its cross-attention to the 512-row text context, and the
text encoder's causal attention with an F16 mask. Unfused, one DiT pass took
29.7 s on the NPU; fused, 0.89 s, with the same velocity.

Against the PyTorch dumps (`test-moss-sfx-parity` with
`MOSS_SFX_BACKEND=hexagon`), cosine similarity per stage:

| GGUF | Text encoder | DiT, prompt / empty | VAE decode | Eight guided steps |
|---|--:|--:|--:|--:|
| `f16` | 0.999999 | 1.000000 / 1.000000 | 0.999994 | 0.999998 |
| `q8_0` | 0.999687 | 0.999698 / 0.999656 | 0.999994 | 0.999645 |

Speed on the Galaxy S25 (seed 7, 6 CPU threads, each run started at thermal
status 0; "glass" and "rain" are the prompts of the parity dumps and of
[Run](#run-1)):

| Request | GGUF | Backend | Steps | Diffusion | VAE decode |
|---|---|---|--:|--:|--:|
| glass, 5 s | `f16` | Hexagon | 2 | 3.6 s | 1.3 s |
| same | `q8_0` | Hexagon | 2 | 3.2 s | 1.1 s |
| same | `f16` | CPU | 2 | 163.4 s | 9.6 s |
| same | `q8_0` | CPU | 2 | 122.0 s | 7.6 s |
| glass, 5 s | `f16` | Hexagon | 100 | 367.6 s | 2.4 s |
| rain, 10 s | `f16` | Hexagon | 100 | 316.8 s | 4.9 s |
| same | `q8_0` | Hexagon | 100 | 326.9 s | 4.8 s |
| rain, 30 s | `f16` | Hexagon | 100 | 372.7 s | 14.7 s |

A guided step runs the DiT twice over the full 30-second latent, so the
diffusion cost does not depend on the requested length. From a cool start a
pass costs 0.89 s (`f16`) or 0.79 s (`q8_0`) on the NPU against 41 s and 31 s
on the CPU; over a 100-step clip the NPU throttles to 1.6-1.9 s per pass, so a
default clip takes five to six minutes, where the CPU would need over two
hours. Text encoding takes about 0.1 s. The `f16` GGUF (6.4 GB, 5.8 GB of it
on the NPU) runs on the 12 GB phone.

### Test

`test-moss-sfx` builds a tiny random-weight GGUF in-test and needs no
download: the schedule, Euler step, guidance, prompt cleaning against
upstream outputs, Python-compatible rounding, and noise; the load-time
rejections for metadata bounds, tensor shapes, and tensor types; text-encoder
causality, zero padding, and an f16 feed-forward sum past the half-precision
range; DiT conditioning on text and timestep and the scheduler guard;
windowed VAE decoding against a single window; engine determinism, guidance
1, cancellation from the callback and from another thread, overlapping calls,
and request validation; and the CLI's `sfx` and `tts` modes.
`test-convert-moss` also fabricates a tiny MOSS-SoundEffect pipeline and
checks the tensor census, the folded weight norm, the f16 and q8_0 paths, and
the `.pth` VAE checkpoint (when torch is installed). `test-moss-sfx-parity` is the reference check: it
skips unless `MOSS_SFX_MODEL` points at a converted GGUF and
`MOSS_SFX_REFERENCE_DIR` at stage dumps from the PyTorch pipeline
(`MOSS_SFX_GPU=1` runs it on the GPU, `MOSS_SFX_BACKEND=hexagon` on the NPU). The dumps come from
`scripts/dump-moss-sfx-reference.py <checkpoint> <out_dir> --upstream
<MOSS-TTS checkout>`, whose defaults are the prompt, duration, and step count
the test expects. `test-moss-sfx` also covers the sidecar routing without
Core ML: the fixed VAE window plan, sidecar and fallback decodes, retiring a
failed DiT or VAE sidecar, strict mode, the default compute units, and the
per-request backend report. `test-moss-hexagon-layout` forces the Hexagon
layout onto the CPU at f32, f16 and q8_0: the host token table gives the same
text context as the device table, and fused attention in the text encoder and
in the DiT's self- and cross-attention matches the unfused graphs; it also
checks that a host table leaves the device weights and is priced as host
memory. On a phone, `test-moss-hexagon` runs the text encoder, the DiT and the
VAE on HTP0 against the CPU at f32, f16 and q8_0, checks that bf16 weights are
refused, and projects the fit against host memory.
`test-moss-sfx-coreml-exporter` (Python; skips without numpy or torch) checks
the exporter's DiT and VAE rebuilds against independent NumPy references on
tiny random weights, including the reordered rotary pairs, the stride-phase
transposed convolution, the Snake polynomial and the capped `1 / alpha`; with
coremltools it also checks each converted program's inputs and output.

## MOSS-Speech

[MOSS-Speech](https://huggingface.co/fnlp/MOSS-Speech) answers a spoken
question with speech, without a text step in between. A 9B Qwen3 language
model shares its first 32 layers between two channels, then splits into a
4-layer text branch and a 4-layer audio branch, each with its own KV cache,
final norm, and head. Each position carries a (text, audio) token pair: the
text channel reads the text embedding, and a text placeholder hands the
position to the audio embedding. The companion
[codec](https://huggingface.co/fnlp/MOSS-Speech-Codec) turns user speech into
tokens with a causal Whisper-VQ encoder (12.5 tokens per second, 16384 codes)
and speaks the reply with the CosyVoice2 flow-matching decoder and HiFT
vocoder at 24 kHz, conditioned on a voice prompt and a CAM++ speaker
embedding. The decoder is the same S3Gen stack Chatterbox uses, so the engine
reuses it; the codec only changes the token-to-mel ratio and the speech
vocabulary, which the S3Gen loader now reads from the GGUF.

Speech replies with the LM on a GPU or on the
[Hexagon NPU](#hexagon-npu-speech-to-speech) run the LM and S3Gen decoder in
separate memory phases. After token generation, the engine releases LM weights
and KV/graph buffers before decoding audio, while retaining tokenizer metadata
and backend identity. A GPU decoder holds a matched S3Gen cache reference only
for that decode; other cache owners retain their references. The next request
reloads LM weights from the same GGUF, so keep the model file available. This
reduces peak GPU memory, and on a 12 GB phone keeps the request within the
host's memory, at the cost of loading weights again between spoken replies. An
LM on the CPU keeps its resident weights.

**Status — CPU and Metal, validated against the reference pipeline.** Every
stage matches the PyTorch pipeline on a real question: the log-mel features
(cosine 1.0) and the 45 user speech tokens (45 of 45), the voice-prompt mel
(1.0) and the CAM++ embedding (0.99991), the reply mel from the flow decoder
with the reference noise (0.99977), and the prefill logits of both heads
(text 0.99997, audio 0.999998 in bf16; 0.99994 and 0.999998 in q8_0). With
teacher forcing, greedy decoding picks the reference's audio code at 93.5 %
of the positions in both bf16 and q8_0; the misses are near-ties between
bf16 on MPS and ggml, so free-running greedy replies drift apart after a few
tokens: in bf16 the greedy reply still ends on its own, while in q8_0 it can run
on to the token limit, so keep the default sampling for q8_0. Sampled replies to English questions are
coherent and intelligible in both quantizations. On an M1 Ultra with Metal
the LM runs at about 32 tokens per second in bf16 and 47 in q8_0, faster than
the 12.5 tokens per second of speech it produces; while it speaks, each step
skips the text head, which the audio channel does not read.

### Convert

```sh
# language model: shared trunk, both branches, both heads, tokenizer,
# special tokens, and the default system prompts
python scripts/convert-moss-speech-to-gguf.py /path/to/MOSS-Speech \
    --outtype q8_0 --outfile moss-speech-q8_0.gguf

# codec: Whisper-VQ encoder + CosyVoice2 flow + HiFT + CAM++ + default voice
python scripts/convert-moss-speech-codec-to-gguf.py /path/to/MOSS-Speech-Codec \
    --campplus-gguf cosyvoice3-campplus-f32.gguf \
    --voice-wav /path/to/MOSS-Speech/assets/prompt_en.wav \
    --outtype f16 --outfile moss-speech-codec-f16.gguf
```

The LM converter writes `bf16` (18.2 GB, the default), `q8_0` (9.7 GB),
`f16`, and `f32`; norms stay f32. For a 12 GB phone, quantize the transformer
layers of the `f16` file to `q4_0` with `requantize-gguf.py` (see
[Hexagon NPU](#hexagon-npu-speech-to-speech)); the loader takes `q4_0` weights
on every backend. The codec converter needs torch,
safetensors, transformers, librosa, and soundfile, and takes the CAM++
weights from the CosyVoice3 CAM++ GGUF (the same network the CosyVoice2
frontend runs). `--voice-wav` becomes the default reply voice, stored as mono
float samples at its original rate (at most 60 s). The codec's `--outtype` is
`f16` (1.3 GB, validated), `q8_0`, or `f32`; the conv kernels and the
Whisper-VQ matrices stay f16 in the quantized file.

### Run

```sh
build/moss-cli --mode s2s --model moss-speech-q8_0.gguf \
    --codec moss-speech-codec-f16.gguf --audio question.wav \
    --gpu --out reply.wav
```

`--audio` is the user turn (any rate from 8 to 192 kHz, stereo downmixed).
`--voice ref.wav` (up to 60 s) replaces the default voice, `--system "..."` adds a system
turn, and `--text-reply` asks for a text answer instead of speech (printed to
stdout). Sampling defaults to the upstream values: temperature 0.7, top-p
0.95, top-k 20, seed 0; `--greedy`, `--temperature`, `--top-p`, `--top-k`, and
`--seed` override them. By default, generation continues until the model ends
its reply or the remaining model context is exhausted. `--max-new-tokens N`
sets an explicit reply budget (`0`, the default, uses the remaining context),
and `--max-reply-seconds` cuts the spoken part cleanly by forcing
the end-of-speech code. There is no streaming in this mode. `--backend NAME`
places the LM and the speech tokenizer on one device (`cpu`, `hexagon`, or an
exact ggml device name) with no fallback; `--gpu` then places the S3Gen decoder
on the automatic GPU, and without it the decoder runs on the CPU. Without
`--backend`, `--gpu` places all three on the automatic GPU.

### Engine notes

The public API is `tts_cpp::moss::SpeechEngine` in
[`include/tts-cpp/moss/speech.h`](../include/tts-cpp/moss/speech.h):
construct with `SpeechOptions` (LM and codec paths, threads, `use_gpu`,
`backends_dir`, `backend`) and call `respond(request, progress)` with a `SpeechRequest`
that holds the conversation as `SpeechMessage`s. Each message carries either
text or audio (PCM plus sample rate), roles are system, user, and assistant,
and the last message must be the user's. The `SpeechResult` holds the reply
PCM at 24 kHz, the text channel's output, token counts, whether the reply was
truncated by `max_reply_seconds` or `max_new_tokens`, and the encode, prefill, generation, and decode wall times.
`cancel()` from another thread stops a run at the next speech-tokenizer
segment, prefill batch, generation step, or decoder step, and `false` from the
progress callback stops generation; one response runs at a time per
instance. A request holds up to 256 messages, each with up to 16384 bytes of
text or 600 s of audio.

The prompt follows the upstream processor: each turn is
`<|im_start|>role\n` + content + `<|im_end|>\n`, a spoken turn is the
speech-start token, one placeholder row per speech code, and the
end-of-speech code, and when the conversation does not open with a system turn the default
one ("Respond with spoken outputs", or with text outputs for `--text-reply`)
is appended after the turns, as upstream does. The spoken reply opens with
the assistant header and the speech-start token. Generation mirrors the
reference loop: the channel switches to audio after speech-start and back to
text after end-of-speech, audio codes past end-of-speech are masked, the
end-of-speech code is masked for the first nine tokens (upstream's
`min_new_tokens` of 10 counts from one) and `<|im_end|>` on the text channel
for the first ten, as Hugging Face's minimum-length processor does, the same temperature
and top-k/top-p apply to both heads, the text channel is forced to the
placeholder while speaking, and generation stops on `<|im_end|>` or the pad
token. The reply's speech codes are the audio channel from the first
speech row up to the first end-of-speech. A stop row closes the reply and is
dropped; a reply cut by `max_new_tokens` keeps its last row.

The voice prompt is prepared the way the CosyVoice2 frontend does it: speech
tokens from the Whisper-VQ encoder, an 80-band 24 kHz mel trimmed to four
frames per token, and a CAM++ embedding of the 16 kHz audio. The default
voice is prepared once per engine and cached. CAM++ runs on the scalar CPU
path, which is the one that matches the reference; its convolutions now
accumulate along time rows so the compiler vectorizes them, with the same
summation order per output, which cuts the 36-second default voice from
about 24 s to about 4 s on an M1 Ultra. The flow decoder's source noise in
HiFT is random, so the reply waveform is compared with the reference through
its log-mel spectrum. The LM loader validates the geometry, the special
tokens, and every tensor's shape before it allocates; the KV cache is sized
from the prompt plus `max_new_tokens`, rounded up to 256 positions. With the
default `max_new_tokens` of 0 that is the whole 40,960-position context, 6.7 GB
of KV cache. On a backend that shares the host's memory (CPU, integrated GPU,
Metal, Hexagon) `begin()` refuses a cache larger than the memory the host
reports free, naming both sizes, instead of letting the allocation exhaust the
device; set `max_new_tokens` to bound it.

### Core ML speech tokenizer

On Apple builds with `TTS_CPP_COREML=ON`, the Whisper-VQ encoder that turns the
user's speech and the voice prompt into speech tokens can run as a Core ML
model; the vector quantizer, the LM, the flow decoder, HiFT, and CAM++ stay on
ggml.

```sh
python scripts/export-moss-speech-tokenizer-coreml.py \
    --gguf moss-speech-codec-f16.gguf --compile-dir .
```

The engine looks for `<stem>-tokenizer.mlmodelc` beside the codec GGUF, with
any quantization tag dropped (`moss-speech-codec-tokenizer.mlmodelc`). The
sidecar takes one full 30 s segment, the tokenizer's segment length, and
returns the pooled encoder states; the nearest-code search stays on ggml. Only
full segments go to the sidecar: the ggml encoder runs just the frames a
shorter segment covers, which is faster than a full Core ML segment, so a
short question, and the last part of a long turn, stay on ggml. A failed call
retires the sidecar and the segment and the rest of the request encode on
ggml.
`SpeechEngine::tokenizer_on_coreml()` says whether a sidecar is attached, and
`SpeechResult::tokenizer_backend` reports where the request's segments were
encoded (`ggml`, the Core ML placement label, or `mixed` when segments ran on
both); the CLI appends it to the encode time. `MOSS_COREML_DISABLE`, `MOSS_COREML_STRICT`, and
`MOSS_COREML_COMPUTE_UNITS` work as for the
[SoundEffect sidecars](#core-ml-dit-and-vae).

Like the other tts MOSS sidecars, the tokenizer defaults to `cpu_and_gpu`. On
the Neural Engine (`all` or `cpu_and_ane`) its fp16 states land on a
different nearest code for 6-8 % of the tokens, below the 95 % agreement
floor of the reference test; on the GPU 99.2-99.7 % of the codes match the
ggml tokenizer. `bench-moss-coreml` against the ggml tokenizer on Metal,
f16 codec, median of five after a warm-up, measured 2026-10-08:

| Machine | ggml Metal | `cpu_and_gpu` (default) | `all` |
|---|---:|---:|---:|
| Mac mini M4 | 495 ms | 405 ms (1.24x) | 348 ms (1.42x) |
| M3 Ultra | 105 ms | 96 ms (1.09x) | 179 ms (0.59x) |

Each row is one full 30 s segment. A full Core ML segment costs the same
whatever the audio covers, while ggml's cost follows the audio: a 5 s segment
takes 63 ms on ggml and took 390 ms padded on the M4 sidecar, and the two
break even only past roughly 80-90 % of a segment on both machines. That is why only full
segments use the sidecar; typical short questions never reach it.

### Hexagon NPU speech-to-speech

MOSS-Speech runs its language model and speech tokenizer on the Snapdragon
Hexagon NPU (HTP0; measured on a Galaxy S25, Snapdragon 8 Elite, Hexagon v79)
when requested explicitly: `SpeechOptions::backend = "hexagon"` or `moss-cli
--mode s2s --backend hexagon`. As for MOSS-TTS there is no fallback:
construction fails when HTP0 is missing. The reply decoder (the S3Gen flow
decoder and HiFT shared with Chatterbox) does not run on the NPU: add `--gpu`
(`use_gpu`) to decode on the Adreno GPU through OpenCL, otherwise it runs on the
CPU. CAM++ always runs on the CPU. It needs a `qvac-ext-ggml@speech` build with the Hexagon
backend; the generated-model tests also pass without the DSP buffer remapping
fix ([tetherto/qvac-ext-ggml#124](https://github.com/tetherto/qvac-ext-ggml/pull/124)),
and the full-model measurements below were taken with it.

Weights go into the Hexagon repack buffer type, and HTP runs `f32`, `f16`,
`q8_0` and `q4_0` weights. The LM refuses other types at load, including the LM
converter's default `bf16`; the tokenizer, whose loader takes `f32`, `f16`,
`bf16` and `q8_0`, refuses `bf16` (the codec converter writes its matrices at
`f16`). The text embedding table, the audio
embedding table and the text head stay on the host, mapped from the GGUF as
for [MOSS-TTS](#hexagon-npu): at 151,680 x 4,096 the text table and head are
1.24 GB each at `f16`, past HTP's 1 GiB buffer and its 32,768-row limit for
quantized heads. The text head runs on the CPU only on text-channel steps;
while the reply is spoken each step skips it. The audio head stays on the NPU.
Attention runs as one `FLASH_ATTN_EXT`: per layer in the LM, whose value
caches then use the key layout, with an F16 causal mask on prefill batches
(a single decoding row attends to the whole cache without one), and per layer
in the Whisper-VQ encoder with an F16 causal mask.

The 9B LM is 18 GB at `f16`. On a 12 GB phone, quantize the transformer
layers to `q4_0` and keep the tables and heads at `f16`:

```sh
python scripts/requantize-gguf.py moss-speech-f16.gguf \
    moss-speech-blk-q4_0.gguf q4_0 --name-filter blk.
```

The file is 7.1 GB, about 4.3 GB of it on the NPU. Bound every request with
`max_new_tokens` (`--max-new-tokens`): the default sizes the KV cache for the
whole 40,960-position context (6.7 GB), which `begin()` refuses there with the
sizes it needed and found. Even bounded, the LM, the tokenizer and the decoder
held together left about 0.5 GB of the phone's memory free, so the engine
releases the LM weights before each decode, releases the decoder after it, and
reloads the LM on the next request: at least 3.0 GB then stays free, and the
reload adds about 4.6 s to the next request's prefill.

Against the PyTorch dumps (`test-moss-speech-parity` with
`MOSS_SPEECH_BACKEND=hexagon`; same question as the reference, CUDA bf16 dumps):

| Stage | `q4_0` transformer, Hexagon | same GGUF, CPU (x86) | `f16`, CPU (x86) |
|---|--:|--:|--:|
| user speech tokens | 52 / 53 | 53 / 53 | 53 / 53 |
| prefill text logits (cosine) | 0.99359 | 0.99405 | 0.99995 |
| prefill audio logits (cosine) | 0.99995 | 0.99995 | 0.999998 |
| teacher-forced audio codes | 78.5 % | 76.8 % | 92.8 % |

The voice-prompt mel and the CAM++ embedding (cosine 1.000000) and the reply
spectrum (0.999576 on the CPU and on OpenCL) do not depend on the LM's
backend. `q4_0` transformer layers fall below the test's 90 % teacher-forced
floor on every backend, so the parity test reports that stage as a failure for
this tier; the Hexagon LM follows the CPU's numbers for the same file. Sampled
replies stay coherent: Whisper transcribes the S25 reply below as "The tallest
mountain in the world is Mount Everest. It stands at about 10.89 kilometers
above sea level, which is roughly 35,500 feet. ..." (the facts are the model's),
the same with either decoder.

Speed on the Galaxy S25 (the 4.2 s question "What is the tallest mountain in
the world and roughly how tall is it?", seed 1, `--max-new-tokens 600`, 6 CPU
threads, each run started at thermal status 0; decode includes preparing the
default voice on the first request):

| LM GGUF | LM and tokenizer | Decoder | Reply | Encode | Prefill (80 rows) | Generate | Decode |
|---|---|---|--:|--:|--:|--:|--:|
| transformer `q4_0` | Hexagon | CPU | 15.8 s | 0.11 s | 0.42 s | 19.9 s | 356 s |
| transformer `q4_0` | Hexagon | OpenCL | 15.8 s | 0.11 s | 0.39 s | 21.0 s | 63 s |
| all `q4_0` | CPU | OpenCL | 9.6 s | 0.73 s | 4.99 s | 13.3 s | 179 s |

The NPU encodes the question 6.8x and prefills 12x faster than the CPU. A
generation step costs about 100 ms on the NPU and 110 ms on the CPU: both
stream the 4.3 GB of `q4_0` layers per step, so generation stays slower than
real time (a step is 80 ms of audio). The decoder dominates the request: the
S3Gen decoder took 350 s for the 15.8 s reply on the CPU and 56 s on OpenCL.
The 7.1 GB transformer-`q4_0` file would not fit beside the codec on the
phone's CPU, so the CPU row uses the all-`q4_0` file (5.1 GB, heads quantized
too) as its baseline.

### Test

`test-moss-speech` needs no download: it builds a tiny random-weight LM and
a tiny Whisper-VQ GGUF in-test and covers the generation state machine
(channel switches, masked codes, the minimum length, the reply-length cut,
the stop tokens, and reply code extraction), the prompt layout against the
processor's grid, the reply text, the LM load-time rejections, batched
prefill against one batch and against cached steps, the embedding selection
per channel, context overflow, the tokenizer's segment arithmetic, log-mel
normalization, per-segment encoding and its rejections, request validation,
and the CLI's `s2s` flags. `test-convert-moss` fabricates a tiny MOSS-Speech
checkpoint and checks the tensor census, the metadata, the token types, all
four output types, and the rejections, and exercises the codec converter's
Whisper-VQ, CAM++, and default-voice helpers when torch is installed.
`test-moss-speech-parity` is the reference check: it skips unless
`MOSS_SPEECH_MODEL` and `MOSS_SPEECH_REFERENCE_DIR` point at the converted LM
and at dumps of the PyTorch pipeline, and it adds the tokenizer and decoder
stages when `MOSS_SPEECH_CODEC` points at the codec GGUF (`MOSS_SPEECH_GPU=1`
runs it on the GPU, `MOSS_SPEECH_BACKEND=hexagon` runs the LM and the tokenizer
on the NPU, `MOSS_SPEECH_DUMP_MEL=<file>` writes the reply mel). It
prints every number in the status paragraph above and fails below backend-safe
floors: cosine 0.9999 for the user log-mel, 0.999 for the voice mel and the
CAM++ embedding, 0.99 for the reply spectrum and the prefill logits, 95 % of
the user speech tokens, and 90 % teacher-forced agreement. The dumps come from
`scripts/dump-moss-speech-reference.py <MOSS-Speech> <MOSS-Speech-Codec>
<out_dir> --upstream <MOSS-Speech checkout> --user-wav question.wav --device
mps`, which writes the `lm/` and `codec/` folders the test reads.

`test-moss-speech` also loads `q8_0` and `q4_0` LMs and refuses a `q5_0` one,
and checks the KV-cache memory guard. `test-moss-hexagon-layout` forces the
Hexagon layout onto the CPU at f32, f16 and q8_0: the host tables give the same
heads as the device tables, before and after the weights are released and
reloaded, fused attention in the LM matches the unfused graph in batched prefill,
decoding and audio-only steps, the fused Whisper-VQ encoder gives the same codes,
and `use_gpu` places the decoder on the GPU beside an explicit backend. On a
phone, `test-moss-hexagon` runs the LM (prefill in batches, decoding, audio-only
steps, weight eviction) and the tokenizer on HTP0 against the CPU at f32, f16,
q8_0 and q4_0, and checks that bf16 LM and tokenizer weights are refused, the
tokenizer's refusal without offering `q4_0`. The generated GGUFs are narrow:
only the LM's 32-wide feed-forward down projection holds a quantization block,
so the quantized tiers leave the tables, the other LM matrices and the whole
tokenizer at `f16`; the full checkpoint covers `q4_0` layers on the NPU.
`test-moss-speech-e2e hexagon` makes two replies on one engine with the LM on
the NPU, and `test-moss-speech-parity` checks that a decoder released after its
decode leaves no S3Gen graph cache.

`test-moss-speech` also covers the tokenizer sidecar's routing without Core
ML: the window size, the sidecar path, full segments on the sidecar and partial
ones on ggml, a failed sidecar that is retired, strict mode, and the backend
report. `test-moss-speech-tokenizer-coreml-exporter`
(Python; skips without numpy or torch) checks the exporter's causal Whisper-VQ
rebuild against an independent NumPy encoder and its causality, and, with
coremltools, the converted program's input and output. `test-moss-coreml-parity` is the on-device check for all four tts
sidecars and runs on a Mac with compiled sidecars beside the GGUFs:
`MOSS_SPEECH_COREML_CODEC`, `MOSS_SFX_COREML_MODEL`, and
`MOSS_CODEC_COREML_DECODER` select the stages (a stage without its variable is
skipped), and `MOSS_COREML_PARITY_GPU=1` runs the ggml side on the GPU. It
compares each sidecar with ggml (cosine 0.999 for the DiT velocity, the VAE
and codec audio; 95 % of the speech tokens on one and two full segments), and
covers attachment, the tokenizer's default GPU placement and its partial
segments on ggml, the codec's streams on the sidecar and batch decodes on ggml,
the disable and strict switches, and an invalid sidecar. `bench-moss-coreml` takes the same
variables and prints each stage's time on ggml and on Core ML.

## Licenses

The [MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5),
[MOSS-TTSD-v1.0](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0), and
[MOSS-Audio-Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer)
model cards identify Apache-2.0 weights. The reference
[MOSS-TTS repository](https://github.com/OpenMOSS/MOSS-TTS/blob/main/LICENSE)
is also Apache-2.0. See [NOTICE](../NOTICE) for the other MOSS models and codec sources.
