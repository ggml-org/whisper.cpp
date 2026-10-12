# parakeet engine: MOSS-Transcribe-Diarize

Part of the [parakeet engine documentation](../README.md).

CUDA or Vulkan can be selected explicitly with `--backend cuda` or
`--backend vulkan` (or an exact device such as `--backend CUDA0`). This overrides `--gpu` and fails when
the requested backend is unavailable; `--backend auto` retains the existing
`--gpu`/CPU selection. The API equivalent is `TranscribeOptions::backend`.
Build against a CUDA- or Vulkan-enabled ggml installation, then run:

```sh
build/moss-transcribe --model moss-transcribe-f16.gguf --audio speech.wav --backend cuda
ctest --test-dir build -R '^test-moss-transcribe-(cuda|vulkan)$' --output-on-failure
```

Each arm registers only when ggml carries its backend. The test generates random-weight GGUFs in F32, F16 and mixed Q8_0/F16
and compares the encoder, adaptor, batched prefill and incremental decoding
against CPU. It also checks explicit selection through the public engine and
rejects unavailable backends. It runs in the manual `parakeet CI` GPU lane
(`run_gpu=true`) without model downloads. These checks pass on Apple M2
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
OpenCL and CPU-repacking backends rebuild their layout from the full tensor.
Timestamps are parsed and JSON numbers written in the C locale, so a host
application with a comma-decimal locale gets the same segments and valid JSON.

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
JSON output, the timestamp and JSON numbers under a comma-decimal locale, and
the weight-upload plan per buffer kind. `test-converter-moss-transcribe` fabricates a tiny checkpoint
and checks the tensor census, the column biases, the token types, the stored
prompt ids, the hotword metadata, and the q8_0 path (skipped without
`numpy`, `gguf`, or `tokenizers`).
`test-moss-transcribe-parity` is the reference check: it skips unless
`MOSS_TRANSCRIBE_MODEL` points at an f16 GGUF (a quantized model passes the
stage checks but its greedy transcript drifts from the fp32 one, although it
scores the same WER) and
`MOSS_TRANSCRIBE_REFERENCE_DIR` at stage dumps from the Hugging Face model
(`MOSS_TRANSCRIBE_GPU=1` runs it on the GPU). The dumps come from
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
CPU and unified-memory GPU verdicts charge host memory to the same pool. Workloads past
the model context return `Error` / `workload-too-large`; invalid requests return
`Error` / `invalid-arguments`.

`test-moss-transcribe-fit` checks weightless/full-file parity, workload scaling, invalid
arguments, saturating margins and actual CPU allocation parity on a tiny GGUF. Set
`MOSS_TRANSCRIBE_MODEL` for a real-model projection and `MOSS_TRANSCRIBE_GPU=1` to
require the GPU backend.
