# parakeet engine: MOSS-Transcribe-Diarize

Part of the [parakeet engine documentation](../README.md).

CUDA, Vulkan, OpenCL, Metal or the Snapdragon Hexagon NPU can be selected explicitly
with `--backend cuda`, `--backend vulkan`, `--backend opencl`, `--backend metal` or
`--backend hexagon` (or an exact device such as `--backend CUDA0`). This overrides `--gpu` and fails when
the requested backend is unavailable; `--backend auto` retains the existing
`--gpu`/CPU selection. The API equivalent is `TranscribeOptions::backend`.
Hexagon and Adreno OpenCL have their own graph layouts; see
[Hexagon NPU](#hexagon-npu) and [Adreno OpenCL](#adreno-opencl).
Build against a CUDA-, Vulkan-, OpenCL- or Hexagon-enabled ggml installation,
then run:

```sh
build/moss-transcribe --model moss-transcribe-f16.gguf --audio speech.wav --backend cuda
ctest --test-dir build -R '^test-moss-transcribe-(cuda|vulkan|hexagon|opencl|metal)$' --output-on-failure
```

Each arm registers only when ggml carries its backend. The test generates random-weight GGUFs in F32, F16 and mixed Q8_0/F16
and compares the encoder, adaptor, batched prefill and incremental decoding
against CPU. It also checks explicit selection through the public engine and
rejects unavailable backends. The CUDA and Vulkan arms run in the manual
`parakeet CI` GPU lane (`run_gpu=true`) without model downloads; the Hexagon
and OpenCL arms run on a Snapdragon device. The Metal arm registers in a Metal build
and runs on a Mac; no CI lane runs it yet. The generated models are too small
for the Adreno kernels described under [Adreno OpenCL](#adreno-opencl), so
that path is covered by the reference parity test. These checks pass on Apple M2
through MoltenVK; full-checkpoint transcript quality and discrete-GPU/mobile
validation remain separate from this generated-model coverage.

[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
transcribes speech and labels who spoke when, in one pass over the whole
file. A Whisper-medium-shaped encoder (fine-tuned weights) turns each 30 s
window of 16 kHz audio into 1500 frames, a 4x temporal-merge adaptor brings
them to 12.5 tokens per second, and a 0.6B Qwen3 decoder reads those tokens
in place of `<|audio_pad|>` placeholders and writes the transcript as
`[start][Sxx]text[end]` segments. It ships in the parakeet engine next to the
other speech-to-text and diarization models, with its own API and CLI; it
does not go through `parakeet::Engine`.

**Status — CPU and Metal, validated against the reference model.** On a
two-minute Spanish reference, every stage matches the Hugging Face model in
fp32: the log-mel front end (cosine 1.000000), the encoder (0.99998 on Metal,
0.9997 on CPU), the adaptor (0.999998), and the prefill logits (1.000000);
the greedy transcript is identical token for token (726 tokens, 20
segments). On a long-form set of concatenated FLEURS clips (2, 5, 10, and 30
minutes), the f16 and q8_0 GGUFs on Metal score the same WER and CER as the
reference model: 2.1-3.9% WER in Spanish and 9.4-12.1% CER in Chinese, with
full coverage; q5_0 stays within 1.5 points of them. English is not usable on that set (46-62% WER, the model skips
whole spans), and the reference model does the same.

## Convert

One GGUF holds the encoder, the adaptor, the decoder, the Slaney mel filter
bank, the tokenizer, and the prompt defaults.

```sh
python engines/parakeet/scripts/convert-moss-transcribe-to-gguf.py /path/to/MOSS-Transcribe-Diarize \
    --outtype f16 --outfile moss-transcribe-diarize-f16.gguf
```

`--outtype` picks the type of the linear weights and of the token embedding
(which doubles as the tied output head): `f16` (1.8 GB), `q8_0` (1.0 GB), and
`q5_0` (0.64 GB); `bf16` and `f32` are also written. Rows too short for a
quantization block fall back to f16. Norms, biases, the positional table, and
the mel filters stay f32, and the two encoder convolution kernels stay f16
because ggml's CPU im2col requires it. The converter needs the `tokenizers`
package: it stores the token ids of the default (Chinese) prompt computed by
the Hugging Face tokenizer, plus the hotword prefix (`热词提示：`) and separator
(`, `) that upstream's hotword recipe appends to it.

## Run

```sh
build/moss-transcribe --model moss-transcribe-diarize-f16.gguf \
    --audio meeting.wav --gpu --out meeting.json
```

The input must be a 16 kHz WAV (stereo is downmixed by averaging). The CLI
prints one `[start-end] Sxx: text` line per segment and, with `--out`,
writes a JSON document with the raw text, the segments, the token counts,
and the stage timings. `--hotwords "Tether,QVAC,vcpkg"` lists names and
domain terms the audio is likely to contain; they are appended to the
default instruction the way upstream's hotword recipe does it. `--prompt`
replaces the default instruction instead (upstream's `examples/prompts.md`
lists the English and speaker-only variants; a blank prompt keeps the
default) and cannot be combined with `--hotwords`. `--max-new-tokens` raises
the model default of 5120, which is short for recordings past about fifteen
minutes. There is no streaming in this mode. On an M1 Ultra with Metal and
the f16 model, ten minutes of Spanish take about 95 seconds.

## Engine notes

The public API is `parakeet::moss::TranscribeEngine` in
[`include/parakeet/moss_transcribe.h`](../include/parakeet/moss_transcribe.h):
construct with `TranscribeOptions` (model path, threads, `use_gpu`,
`backends_dir`), call `transcribe(pcm, samples, sample_rate, request,
progress)` for a `TranscribeResult` (text, parsed segments, audio, prompt and
generated token counts, encode, prefill, and decode wall times), and
`cancel()` from another thread or `false` from the progress callback to stop
a run. A request leaves `prompt` and `hotwords` empty and `max_new_tokens` at
zero to use the model defaults. Prompts are capped at 8192 bytes.

`TranscribeRequest::hotwords` biases one request toward the listed terms
without touching the model. Each entry is sanitized before it reaches the
prompt: control characters and the characters that frame the transcript
markup or special tokens (`[ ] < > |`) become spaces, whitespace runs
collapse, and the result is trimmed; entries that end up empty and repeated
entries are dropped. A request takes at most 64 hotwords of at most 64 bytes
each, and throws past either bound. On a Spanish recording full of domain
terms, the default prompt misrecognizes all five (`TESER`, `Qback`, `VCPK`,
`para Kit`, `Nematron`) and the hotword request recovers all five (Tether,
QVAC, vcpkg, Parakeet, Nemotron) with identical timestamps and speaker
labels; the f16 transcript is identical to the Hugging Face model's for both
prompts. One transcription runs
at a time per instance; overlapping calls throw.

CPU encoder attention uses the fused ggml operator with F32 accumulation,
avoiding the full per-head attention score matrices. GPU encoder graph
selection is unchanged; decoder attention differs only in Metal's single-token
graph (see [Metal decoding](#metal-decoding)). Memory-fit measurement
builds the same fused CPU encoder graph used for inference.

Processing mirrors the reference processor. The audio is cut into
non-overlapping 30 s windows, the last one zero-padded; each window gets a
Whisper log-mel spectrogram (periodic Hann, reflect padding, 80 Slaney bands,
`log10`, clamp to the window maximum minus 8, then `(x + 4) / 4`) and keeps
`(samples - 1) / 1280 + 1` tokens. The prompt is the chat template with the
audio span between `<|audio_start|>` and `<|audio_end|>`; every five seconds
the span carries the elapsed seconds as digit tokens, at the placeholder
positions upstream uses. Prompt text is tokenized by a byte-level BPE that
reproduces the Qwen2 pre-tokenizer regex with Unicode letter, number, and
whitespace classes (tables generated by `scripts/gen-moss-unicode-tables.py`),
so Chinese punctuation, full-width forms, and newline runs split as in the
Hugging Face tokenizer. The decoder runs a single greedy pass up to
`<|im_end|>`, with prefill in 256-token batches and an f16 KV cache (about
112 KiB per token, so a 30-minute file with 16k new tokens needs about 4.4
GB) padded to a multiple of 256 positions. The transcript is decoded with
control tokens skipped and parsed by a port of upstream's streaming
`TranscriptStreamParser`: a segment needs a start timestamp, an `S<digits>`
speaker, text, and an end timestamp no earlier than the start; anything else
stays in the text or is dropped, exactly as upstream does. The loader
validates every tensor's shape and type against the metadata and bounds the
metadata itself, so a malformed GGUF fails with an error instead of aborting
inside ggml. Weights are read straight into host-visible buffers (CPU, and
Metal's shared memory) with no staging copy; on other GPU backends float
tensors stream in 8 MiB chunks and quantized tensors go up whole, because the
OpenCL, Hexagon and CPU-repacking backends rebuild their layout from the full
tensor. Timestamps are parsed and JSON numbers written in the C locale, so a host
application with a comma-decimal locale gets the same segments and valid JSON.

## Metal decoding

A decode step is one token through 28 Qwen3 layers. It is bound by the number
of Metal dispatches and the CPU time spent preparing them, not by weight
bandwidth: before the changes below, f16, q8_0 and q5_0 decoded at the same
speed. On Metal the decoder therefore builds the single-token graph
differently from prefill and from the other backends:

- **One graph per 64-position window.** The step graph writes the new key and
  value through row-index inputs (`ggml_set_rows`) and attends over a
  64-aligned window of the cache behind a mask, so a single built and
  allocated graph serves every step until the window fills. Building the graph
  and allocating it in the scheduler (including Metal's graph reordering) used
  to take about 0.7 ms per token. Positions past the current one are masked,
  and the cache there is still zero, so they contribute nothing.
- **No F16 overflow guard in the feed-forward.** Prefill and the other backends
  scale the gated product by 1/64 before the down projection and by 64 after
  it, because their matrix kernels round activations to f16. Metal's
  matrix-vector kernels read f32 activations, and a power-of-two scale is
  exact there, so the single-token graph drops the two dispatches with
  bit-identical results.
- **Residual added inside the matrix-vector product.** The projection is the
  first operand of the residual add, which lets Metal fuse the add into the
  quantized (q8_0, q5_0) kernels.
- **Grouped query heads.** The two query heads that share a KV head are
  scored as one two-column product, which reads each cached key once and uses
  Metal's small-batch kernel.
- **No copies of contiguous tensors** before the cache write, nor of the
  single-token attention output. This applies on every backend and is exact.

Every greedy token stays the same. On a 65 s English clip, teacher-forced
through the previous graph's tokens, all 368 steps pick the same token on f16,
q8_0 and q5_0, and the logits stay within 0.0045 of the previous graph. The
transcripts of the four clips below are identical to the previous graph's in
all three quantizations. Flash attention was not adopted: on Metal it decoded
no faster than the unfused graph, and it moved the logits further (0.013).

Warm request times on an M3 Ultra (Metal, 8 threads, third of three requests in
one process; clips from `test/samples`), previous graph to the current one:

| Clip | Generated tokens | Decode f16 (ms) | Decode q8_0 (ms) | Decode q5_0 (ms) | Request f16 (ms) |
|---|--:|--:|--:|--:|--:|
| `jfk.wav`, 11 s | 70 | 471 to 334 | 436 to 288 | 438 to 290 | 666 to 530 |
| `diarization-sample-16k.wav`, 27 s | 185 | 1289 to 900 | 1196 to 755 | 1209 to 788 | 1509 to 1119 |
| `abcba.wav`, first 65 s | 367 | 2744 to 1863 | 2531 to 1547 | 2549 to 1652 | 3298 to 2416 |
| `abcba.wav`, 160 s | 943 | 8245 to 5266 | 7720 to 4512 | 7768 to 4804 | 9381 to 6399 |

Decode is 29 to 42% faster, and the gain grows with transcript length. The
encoder and prefill are unchanged. The CPU and the other GPU backends still
build a graph per step; apart from the dropped copies their graphs are the same,
and their output is unchanged.

## Core ML encoder sidecar

On Apple builds with `PARAKEET_COREML=ON`, the audio encoder and the adaptor
can run as a Core ML model while the decoder stays on ggml. The sidecar takes
one 30 s window of log-mel features and returns that window's 375 adaptor
embeddings; the engine keeps the leading rows the window needs. Every window is
already 30 s, the last one zero-padded exactly as the reference processor
pads it, so the sidecar computes the same function as the ggml encoder at
every input length. Export it from any quantization of the GGUF (the encoder
and adaptor weights are dequantized; the default program is fp16):

```sh
python engines/parakeet/scripts/export-moss-transcribe-encoder-coreml.py \
    --gguf moss-transcribe-diarize-f16.gguf --compile-dir .
```

The export environment is the one in
[`scripts/requirements-coreml.txt`](../scripts/requirements-coreml.txt) (see
[backends](backends.md#core-ml-encoder-sidecar)). Conversion runs on Linux
too; `--compile-dir` calls `xcrun coremlcompiler`, so compile on a Mac or
compile the `.mlpackage` there later. `--palettize 8|6|4` writes palettized
weights, `--precision float32` an fp32 program, and `--parity-dir` checks the
PyTorch rebuild against the dumps of
`scripts/dump-moss-transcribe-reference.py` before exporting. The engine looks
for `<stem>-encoder.mlmodelc` beside the GGUF, where the stem drops a trailing
quantization tag, so `moss-transcribe-diarize-q8_0.gguf` and
`moss-transcribe-diarize-f16.gguf` both use
`moss-transcribe-diarize-encoder.mlmodelc`.

A sidecar whose input or output shapes do not match the model is not attached.
If a prediction fails, the engine retires the sidecar, encodes that window and
the rest of the run on ggml, and reports the run as `mixed`.
`TranscribeEngine::encoder_on_coreml()` says whether a sidecar is attached, and
`TranscribeResult::encoder_backend` says where this run's windows went: `ggml`,
the Core ML placement label (`coreml-all`, `coreml-cpu-gpu`, `coreml-ane`,
`coreml-cpu`), or `mixed`. The CLI prints the placement on stderr and writes
`encoder_backend` to the JSON. Three environment variables control routing:
`MOSS_COREML_DISABLE` keeps the encoder on ggml, `MOSS_COREML_STRICT` makes a
missing sidecar or a failed prediction an error instead of a fallback, and
`MOSS_COREML_COMPUTE_UNITS` (`all`, the default, `cpu_and_gpu`, `cpu_and_ane`,
or `cpu_only`) picks the Core ML compute units.

`bench-moss-transcribe-coreml` times one 30 s window of
`test/samples/diarization-sample-16k.wav` with the f16 GGUF (median of five
after a warm-up) against the ggml encoder on Metal, measured 2026-10-08:

| Machine | ggml Metal | `all` (default) | `cpu_and_gpu` | `cpu_and_ane` |
|---|---:|---:|---:|---:|
| Mac mini M4 | 520 ms | 290 ms (1.79x) | 396 ms (1.31x) | 291 ms (1.79x) |
| M3 Ultra | 96 ms | 169 ms (0.57x) | 86 ms (1.11x) | 307 ms (0.31x) |

On the M4 the default placement runs the encoder on the Neural Engine; its
embeddings match the ggml encoder at cosine 0.99964 (worst token 0.9948) and
the greedy transcript is identical. On the GPU they match at 0.999999. The
encoder is a small part of a transcription: on the M4 the 27 s sample takes
3.51 s with the sidecar and 3.71 s without, because the decoder dominates. On
an M3 Ultra the GPU outruns the Neural Engine, so the default placement is
slower than ggml there; set `MOSS_COREML_COMPUTE_UNITS=cpu_and_gpu` on such
machines, or leave the sidecar out.

## Hexagon NPU

`--backend hexagon` (API: `TranscribeOptions::backend = "hexagon"`) runs the
encoder, the adaptor and the decoder on the HTP0 session of a ggml-speech built
with `GGML_HEXAGON=ON`. On a Galaxy S25 (Snapdragon 8 Elite, Hexagon v79) every
encoder and decoder graph is a single HTP0 split, so no operation falls back
to the CPU. Requirements:

- ggml-speech with `GGML_HEXAGON=ON`, with the `libggml-htp-v*.so` DSP
  skeletons staged next to the host backends. `backends_dir` is prepended to
  FastRPC's `DSP_LIBRARY_PATH` (see [backends](backends.md)). The measurements
  below use qvac-ext-ggml `speech` at `6bb85ad6`, whose HTP `q8_0` activation
  quantizer matches the CPU's.
- A `q8_0` or `f16` GGUF. HTP matrix kernels do not cover `q5_0`, so a `q5_0`
  model is refused at load (`HTP0 cannot project the q5_0 token table`).

The Hexagon graph differs from the other backends in three ways:

- Weights go to the session's repack buffer. HTP refuses quantized matrix
  multiplications whose weights sit in its default host-visible buffer.
- The tied 151,936-row token table is both a `GET_ROWS` table and the output
  head. HTP cannot gather rows from a quantized table and refuses a
  `MUL_MAT` past 32,768 rows, so the engine reads each token's embedding row
  from the GGUF on the host and projects the head in five chunks of at most
  32,768 rows. The table is not held twice. An `f16` table stays whole: HTP
  gathers F16 rows and projects all of them. Any backend that cannot gather
  from the table or project the whole vocabulary takes the same split.
- Each attention is one `FLASH_ATTN_EXT`. Without it, the unfused graph's F32
  matrix multiplications over 1500 x 1500 score tensors took 85% of the DSP
  time of encoding and prefilling the 27 s sample, and the encoder ran slower
  than on the CPU. The KV cache then stores values in the key layout. Other
  backends keep the unfused graph.

Parity on the S25 against the Hugging Face dumps (`test-moss-transcribe-parity`
with `MOSS_TRANSCRIBE_BACKEND`), cosine:

| Stage | CPU f16 | HTP0 f16 | CPU q8_0 | HTP0 q8_0 |
|---|--:|--:|--:|--:|
| Encoder states | 0.99825 | 0.99926 | 0.99333 | 0.99753 |
| Adaptor output | 0.999998 | 0.999999 | 0.99956 | 0.99987 |
| Prefill logits | 0.999999 | 0.999999 | 0.99988 | 0.99996 |
| Greedy transcript | exact | exact | exact | exact |

HTP0 is closer to the reference than the phone's CPU at every stage. Only
`f16` on HTP0 passes the test's 0.999 threshold for the encoder states; the
ARM CPU with either model and `q8_0` on HTP0 fall below it, while the adaptor
output that the decoder reads passes everywhere.

Speed on the same phone (`moss-transcribe`, 6 CPU threads, milliseconds). The
27 s rows are the median of three runs spaced 20 s apart; each long-clip run
started at thermal status 0 (median of three for `q8_0`, one run for `f16`):

| Audio | Model | Backend | Encode | Prefill | Decode | Total | RTF |
|---|---|---|--:|--:|--:|--:|--:|
| 27.3 s, 185 tokens | `q8_0` | CPU | 6,617 | 1,896 | 3,161 | 11,674 | 0.428 |
| | `q8_0` | HTP0 | 354 | 209 | 2,973 | 3,536 | 0.130 |
| | `f16` | CPU | 7,806 | 2,235 | 5,465 | 15,506 | 0.568 |
| | `f16` | HTP0 | 356 | 267 | 4,246 | 4,869 | 0.178 |
| 160.6 s, 943 tokens | `q8_0` | CPU | 60,117 | 17,827 | 34,378 | 112,322 | 0.699 |
| | `q8_0` | HTP0 | 2,103 | 943 | 26,306 | 29,352 | 0.183 |
| | `f16` | CPU | 70,079 | 21,453 | 44,722 | 136,255 | 0.848 |
| | `f16` | HTP0 | 2,164 | 1,194 | 34,415 | 37,773 | 0.235 |
| 191.2 s, 1,157 tokens | `q8_0` | CPU | 69,752 | 22,453 | 45,576 | 137,781 | 0.721 |
| | `q8_0` | HTP0 | 2,465 | 1,109 | 34,156 | 37,729 | 0.197 |
| | `f16` | CPU | 83,345 | 26,156 | 58,556 | 168,057 | 0.879 |
| | `f16` | HTP0 | 2,490 | 1,294 | 45,531 | 49,315 | 0.258 |

HTP0 encodes 19-28x and prefills 9-20x faster than the CPU. Decoding is bound
by memory bandwidth on both: each token reads the decoder weights, the head
and the KV cache, so HTP0 decodes only 1.06-1.33x faster and decoding is most
of its time (per token on the 27 s clip: 8.3 ms of `q8_0` projections, 2.9 ms
of head, 2.7 ms of attention). On all three clips the CPU and HTP0 transcripts have the same words and
speaker labels; segment timestamps differ by at most 0.05 s.

`test-moss-transcribe-hexagon` runs the CPU-against-backend comparison of the
CUDA and Vulkan arms on HTP0 (generated F32, F16 and mixed Q8_0 models:
encoder, adaptor, partial chunk, batched prefill, incremental decoding, the
engine and the memory projection).

## Adreno OpenCL

`--backend opencl` runs the encoder, the adaptor and the decoder on Adreno
OpenCL with the unfused graph. It needs ggml-speech 2026-10-19 or later
(qvac-ext-ggml `speech` at `1bb92ec5`,
[tetherto/qvac-ext-ggml#126](https://github.com/tetherto/qvac-ext-ggml/pull/126)):
older builds run attention batches of 32 or more query tokens through
Adreno matrix kernels that ignore the tails of their 16 by 64 tiles, so on an
unpadded KV length the prefill logits fall to cosine 0.84 against the
reference and the transcript is garbage.

Adreno OpenCL refuses a strided copy source, and the CPU cannot write a cache
that lives in the OpenCL buffer. Writing the transposed value cache therefore
copies a materialized source there; backends that accept the strided copy
keep the single copy.

Flash attention is also correct on OpenCL but slower (prefill 793 ms against
479 ms for a 256-token batch, decoding about 63 ms per token against 13 ms),
so OpenCL keeps the unfused graph.

Parity on a Galaxy S25 (Adreno 830) against the Hugging Face dumps
(`test-moss-transcribe-parity` with `MOSS_TRANSCRIBE_BACKEND=opencl`), cosine:

| Stage | OpenCL f16 | OpenCL q8_0 |
|---|--:|--:|
| Encoder states | 0.99834 | 0.99451 |
| Adaptor output | 0.999993 | 0.99991 |
| Prefill logits | 1.000000 | 0.99988 |
| Greedy transcript | exact | exact |

As on the phone's CPU, the encoder states fall below the test's 0.999
threshold while the adaptor output that the decoder reads passes.

Speed on the same phone (`moss-transcribe`, 6 CPU threads, milliseconds). Each
OpenCL run started at thermal status 0 (median of three for `q8_0` on the 27 s
and 160 s clips, one run otherwise); the CPU and HTP0 rows are those of
[Hexagon NPU](#hexagon-npu):

| Audio | Model | Backend | Encode | Prefill | Decode | Total | RTF |
|---|---|---|--:|--:|--:|--:|--:|
| 27.3 s, 185 tokens | `q8_0` | CPU | 6,617 | 1,896 | 3,161 | 11,674 | 0.428 |
| | `q8_0` | OpenCL | 2,840 | 721 | 3,515 | 7,076 | 0.259 |
| | `q8_0` | HTP0 | 354 | 209 | 2,973 | 3,536 | 0.130 |
| | `f16` | OpenCL | 3,599 | 1,163 | 5,304 | 10,066 | 0.369 |
| 160.6 s, 943 tokens | `q8_0` | CPU | 60,117 | 17,827 | 34,378 | 112,322 | 0.699 |
| | `q8_0` | OpenCL | 18,169 | 3,595 | 31,204 | 52,969 | 0.330 |
| | `q8_0` | HTP0 | 2,103 | 943 | 26,306 | 29,352 | 0.183 |
| 191.2 s, 1,157 tokens | `q8_0` | CPU | 69,752 | 22,453 | 45,576 | 137,781 | 0.721 |
| | `q8_0` | OpenCL | 21,564 | 3,998 | 42,372 | 67,934 | 0.355 |
| | `q8_0` | HTP0 | 2,465 | 1,109 | 34,156 | 37,729 | 0.197 |

OpenCL is 1.6-2.1x faster than the CPU and HTP0 1.8-2.0x faster than OpenCL.
Every OpenCL transcript in the table is identical to the CPU's, byte for byte.

## Test

`test-moss-transcribe` builds a tiny random-weight GGUF in-test and needs no
download: the mixed-radix FFT against a direct DFT, the log-mel
normalization, the chunk and token arithmetic against the reference
processor's counts, the time-marker placement, the prompt layout for the
default and a custom prompt, the pre-tokenizer's Unicode classes, hotword
sanitation and bounds, byte-level decoding with control tokens skipped, the
transcript parser (edge cases and character-by-character streaming), the
load-time rejections, encoder chunks, decoder prefill in uneven batches
against a single batch, audio injection, engine end-to-end, hotword prompts,
cancellation, request validation, the `moss-transcribe` CLI flags and
JSON output, the timestamp and JSON numbers under a comma-decimal locale,
the weight-upload plan per buffer kind, and the two Hexagon layouts on the
CPU: a token table split into head chunks with host lookups reproduces the
whole table's logits exactly in the F32, F16 and mixed Q8_0 fixtures, and
fused attention matches the unfused graph in the encoder and in prefill
batches of 1, 3 and the whole prompt. It also forces the Metal step graph on
the CPU: over 140 decode steps, across two window boundaries, with both
attention layouts and with two KV heads, one reused graph per window matches a
graph built per step and picks the same tokens, and a decoder taking turns with
another at a different window rebuilds its graph without changing a logit. `test-moss-transcribe-metal`
compares the CPU and Metal stages, as the other GPU arms do, plus 70 decode
steps that cross a window. `test-converter-moss-transcribe` fabricates a tiny checkpoint
and checks the tensor census, the column biases, the token types, the stored
prompt ids, the hotword metadata, and the q8_0 path (skipped without
`numpy`, `gguf`, or `tokenizers`).
`test-moss-transcribe-parity` is the reference check: it skips unless
`MOSS_TRANSCRIBE_MODEL` points at an f16 GGUF (a quantized model passes the
stage checks but its greedy transcript drifts from the fp32 one, although it
scores the same WER) and
`MOSS_TRANSCRIBE_REFERENCE_DIR` at stage dumps from the Hugging Face model
(`MOSS_TRANSCRIBE_GPU=1` runs it on the GPU, `MOSS_TRANSCRIBE_BACKEND=hexagon`
or another backend name selects that backend explicitly). The dumps come from
`engines/parakeet/scripts/dump-moss-transcribe-reference.py <checkpoint> <16 kHz wav>
<out_dir> --upstream <MOSS-Transcribe-Diarize checkout>`. When the directory
also holds `tokenizer_cases.bin` from
`engines/parakeet/scripts/dump-moss-transcribe-tokenizer-cases.py <checkpoint>
<out_dir>`, the test checks the native tokenizer against the Hugging Face ids
on prompt-shaped strings (default, English, and hotword prompts, CJK and
full-width punctuation, whitespace and newline runs, digits, contractions,
emoji).

`test-moss-transcribe` also covers the sidecar routing without Core ML: the
sidecar path rule, window and row selection, retiring a failed sidecar, the
strict and disable switches, and the per-run backend report.
`test-export-moss-transcribe-coreml` (Python; skips without `numpy` or
`torch`) checks the exporter's PyTorch rebuild against an independent NumPy
Whisper encoder and adaptor on tiny random weights, the sidecar name rule, and,
with `coremltools`, the converted program's input and output.
`test-moss-transcribe-coreml-parity` runs on a Mac with a compiled sidecar: set
`MOSS_TRANSCRIBE_COREML_MODEL` to the GGUF and pass a 16 kHz WAV. It compares a
full and a short window with the ggml encoder (cosine 0.999 for the window,
0.99 for every token), checks that a transcription runs on the sidecar and
follows the ggml transcript, and covers strict mode and an invalid sidecar.
`bench-moss-transcribe-coreml` (same arguments) prints one window's time on
ggml and on Core ML for the compute units in `MOSS_COREML_COMPUTE_UNITS`.

## Memory-fit preflight

`parakeet::moss::fit_params` in
[`include/parakeet/moss_transcribe_fit.h`](../include/parakeet/moss_transcribe_fit.h)
projects one batch transcription before loading weights:

```cpp
#include <parakeet/moss_transcribe_fit.h>

parakeet::moss::TranscribeOptions options;
options.model_path = "moss-transcribe-diarize-q8_0.gguf";
options.use_gpu = true;
parakeet::moss::TranscribeRequest request;
request.hotwords = {"QVAC", "Tether"};
request.max_new_tokens = 1024;
const double audio_seconds = 90;
const uint64_t margin_bytes = 256ull * 1024 * 1024;
auto fit = parakeet::moss::fit_params(options, request, audio_seconds, margin_bytes);
std::cout << fit.report;
```

Duration and margin are explicit. The request uses the same prompt, hotword and token
rules as transcription; 0 tokens selects the GGUF default. The fitter reads metadata
and the tensor table, accepts a metadata-only copy, and never uploads weight data,
allocates device payloads, or computes a graph. Backend selection matches the engine.

`FitResult` reports weights, the encoder allocation, aligned F16 KV cache, additional
decoder allocation above the encoder, host memory and available device memory. Encoder
and decoder share a scheduler: the two compute fields sum to the peak compute demand.
Host demand includes tokenizer, audio, embeddings, mel preprocessing, decoder buffers,
scheduler storage and the peak CPU execution scratch across encoder and decoder splits
for the configured thread count, including CPU fallback when a GPU is selected;
CPU and unified-memory GPU verdicts charge host memory to the same pool. A Hexagon
session reports no memory of its own, so its verdict uses the CPU device's free
memory and charges the host memory to it too. Workloads past
the model context return `Error` / `workload-too-large`; invalid requests return
`Error` / `invalid-arguments`.

`test-moss-transcribe-fit` checks weightless/full-file parity, workload scaling, invalid
arguments, saturating margins and actual CPU allocation parity on a tiny GGUF. Set
`MOSS_TRANSCRIBE_MODEL` for a real-model projection and `MOSS_TRANSCRIBE_GPU=1` to
require the GPU backend.
