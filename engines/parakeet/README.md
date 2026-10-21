# parakeet.cpp

Parakeet is a pure C++/ggml implementation of NVIDIA FastConformer ASR,
end-of-turn detection, and Sortformer speaker diarization. Inference requires no
Python, PyTorch, NeMo, or ONNX Runtime. A single `parakeet::Engine` loads CTC,
RNN-T, TDT, EOU, Nemotron, or Sortformer GGUFs and selects the implementation
from GGUF metadata. MOSS-Transcribe-Diarize, a Qwen3-based model that
transcribes and labels speakers in one pass (with optional per-request
hotwords), ships alongside with its own `parakeet::moss::TranscribeEngine`
and `moss-transcribe` CLI. `--backend cuda|vulkan` (or
`TranscribeOptions::backend = "cuda"` / `"vulkan"`) explicitly selects a GPU backend; the
request fails if unavailable. Generated-model CPU/CUDA/Vulkan parity and
F16/Q5/Q8 full-checkpoint smoke cases cover this path. Its `parakeet::moss::fit_params` preflight measures
weights, encoder/decoder graphs, KV cache and host memory, including scheduler storage
and CPU scratch for the configured threads, using GGUF metadata only; see
[docs/moss-transcribe.md](docs/moss-transcribe.md).

## Supported checkpoints

| HF repository | Decoder/task | Mel | `d_model × layers` | Vocab | Parameters | GGUF size | Recorded RTF | Languages and notes |
|---|---|---:|---|---:|---:|---|---|---|
| `nvidia/parakeet-ctc-0.6b` | CTC | 80 | 1024 × 24 | 1024 | 600 M | 697 MiB q8_0 / 1.3 GiB f16 | 0.014–0.046 Metal | English; optional Core ML offline encoder |
| `nvidia/parakeet-ctc-1.1b` | CTC | 80 | 1024 × 42 | 1024 | 1.1 B | 1217 MiB q8_0 | 0.026–0.074 Metal | English; optional Core ML offline encoder |
| `ai4bharat/indic-conformer-600m-multilingual` | CTC default; optional RNN-T export | 80 | 1024 × 24 | 5632 + blank | 600 M | ~701 MiB q8_0 / ~373 MiB q4_0 / 1.3 GiB f16 | 0.008 q8_0 Metal / 0.0019 q8_0 Vulkan | 22 Indic languages; optional Core ML offline encoder; requires `--language` or `EngineOptions::language` |
| `nvidia/parakeet-unified-en-0.6b` | RNN-T | 128 | 1024 x 24 | 1024 | 600 M | 707 MiB q8_0 | 0.004 q8_0 Vulkan / 0.028 q8_0 Metal | English; offline full-context encoder with an optional Core ML sidecar; cache-aware streaming at 80/160/560/1040 ms chunks with 0-1040 ms right context |
| `nvidia/parakeet-tdt-0.6b-v3` | TDT | 128 | 1024 × 24 | 8192 | 600 M | 715 MiB q8_0 / 1.34 GiB f16 | 0.006 q8_0 Metal | About 25 languages, with punctuation and capitalization; optional Core ML offline encoder |
| `nvidia/parakeet-tdt-1.1b` | TDT | 80 | 1024 × 42 | 1024 | 1.1 B | 1225 MiB q8_0 | 0.027–0.079 Metal | English only; no punctuation or capitalization; optional Core ML offline encoder |
| `qvac/TranscribePsy-250M-EU-sub` ([GGUF](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub-gguf), [Core ML](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub-coreml)) | TDT | 128 | 640 × 24 | 8192 | 254 M | 291 MiB q8_0 / 17 MiB encoder-less + 186 MiB Core ML | 0.016 q8_0 CPU (8 threads, x86-64) | Eight European languages (nl, en, fr, de, it, pl, pt, es) with punctuation and capitalization, distilled from TDT 0.6B-v3; published GGUF loads as a TDT model; Neural Engine encoder and batched TDT decoder sidecars, see [Neural Engine sidecars](docs/backends.md#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars) |
| `nvidia/parakeet_realtime_eou_120m-v1` | RNN-T + `<EOU>` | 128 | 512 × 17 | 1027 | 120 M | 246 MiB f16 / 132 MiB q8_0 | 0.0052 Vulkan | English ASR and native end-of-turn token; Core ML exact-shape encoder |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | Prompt-conditioned RNN-T | 128 | 1024 × 24 | 13087 | 600 M | ~1.3 GiB f16 | 0.108 CPU | Locale-conditioned ASR; empty language selects `auto`; cache-aware streaming at 80/160/320/560/1120 ms; Core ML exact-shape offline encoder |
| `nvidia/diar_sortformer_4spk-v1` | Sortformer | 80 | 512 × 18 | n/a | 123 M | 263 MiB f16 / 141 MiB q8_0 / 75 MiB q4_0 | 0.0020 Vulkan | Up to four speakers; offline and sliding-history streaming |
| `nvidia/diar_streaming_sortformer_4spk-v2` | Sortformer | 128 | 512 × 17 | n/a | 117 M | 251 MiB f16 / 134 MiB q8_0 / 72 MiB q4_0 | similar to v1 offline | Streaming-trained; sliding-history streaming |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | Sortformer + AOSC | 128 | 512 × 17 | n/a | 117 M | 251 MiB f16 / 134 MiB q8_0 / 72 MiB q4_0 | similar to v1 offline | Audio-Online Speaker Cache preserves slots across long gaps; optional Core ML whole-network diarizer, plus the older exact-shape batch/AOSC encoder sidecars |
| `nvidia/Nemotron-3-Diarization` | Nemotron diarization + AOSC | 128 | 512 × 31 RoPE | n/a | — | 107 MB q8_0 GGUF | 0.0002 q8_0 CUDA / 0.0002 q8_0 Vulkan / 0.0006 q8_0 Metal / 0.0065 q8_0 OpenCL (Adreno 830) / 0.0005 Core ML (Apple M4) | Eight speakers, 10 ms probabilities, native offline and cached streaming; optional Core ML whole-network diarizer |
| `OpenMOSS-Team/MOSS-Transcribe-Diarize` | Whisper-shaped encoder + Qwen3 decoder (text, speakers, timestamps) | 80 | 1024 × 24 encoder, 1024 × 28 decoder | 151936 | 0.9 B | 1.8 GB f16 / 0.98 GB q8_0 / 0.64 GB q5_0 | 0.05–0.39 f16 Metal (2 to 30 min) / 0.13–0.20 q8_0 Hexagon / 0.26–0.36 q8_0 OpenCL (Galaxy S25, 27 s to 3 min) | Multilingual checkpoint, validated on Spanish and Chinese; English long-form skips spans, as in the reference model; per-request hotwords; separate `moss-transcribe` API and CLI; optional Core ML audio encoder; Hexagon NPU and Adreno OpenCL with q8_0 or f16 |

TDT 0.6B-v3 is multilingual with punctuation/capitalization; TDT 1.1B is
English-only without those features. TranscribePsy is a 254M FastConformer-TDT
distilled from TDT 0.6B-v3 for eight European languages and published as
ready-made GGUF and Core ML files; the engine treats it as a TDT checkpoint, and
`--window-sec 60 --overlap 0.25` is the long-form setting validated on phones
(see [CLI](docs/cli.md#long-form-window-flags)). IndicConformer exports CTC by default,
with an optional RNN-T branch; CTC requires a language selection. Use the
[model acquisition and format guide](docs/models.md) for each family's
download and conversion path.

Nemotron 3 Diarization can also be converted to F16, BF16, Q6_K, Q5_K,
Q4_K, MXFP4, and NVFP4 with `scripts/convert_nemotron_diarization.py --outtype`.
F16/BF16 and the K-quants plus NVFP4 pass the CPU/CUDA offline fixture gates;
MXFP4 fails CPU quality checks. Q4_K cuts model size by 46% on the tested
checkpoint. Q4_K/Q6_K run as one GPU graph on Metal and Vulkan and pass
the offline fixtures; their live four-speaker fixture still fails the 15%
DER gate on both backends. NVFP4 passes offline Vulkan validation but uses
CPU matmuls on Metal, so it is not recommended for Metal acceleration.
See [precision-specific quality and timings](docs/performance.md#nemotron-3-diarization-quantizations),
including streaming limitations and reproduction commands.

Unified RNN-T and Nemotron 3.5 ASR have native cache-aware streaming.
[docs/api.md](docs/api.md) documents trained operating points and context
controls. Nemotron 3 Diarization supports eight speakers and uses cached
long-form processing for offline inputs over 90 seconds; see
[its model guide](docs/nemotron-3-diarization-port.md).

## Core ML sidecars

Optional Apple sidecars support eligible CTC, IndicConformer, Unified RNN-T,
TDT, EOU, Nemotron 3.5 ASR and Sortformer v2.1 encoder paths, and the
MOSS-Transcribe-Diarize audio encoder. Input routing
is model-specific; native cache-aware ASR streaming stays on ggml. Missing or
incompatible sidecars fall back to ggml. Export, placement and routing
contracts are in [docs/backends.md](docs/backends.md#core-ml-encoder-sidecar)
and, for MOSS-Transcribe-Diarize, in
[docs/moss-transcribe.md](docs/moss-transcribe.md#core-ml-encoder-sidecar).

TDT models can additionally run on the Apple Neural Engine end to end: a
fixed-window encoder export with padding masks and a batched TDT decoder
sidecar (`<model>-decoder.mlmodelc`). With the decoder sidecar the engine cuts
long recordings at pauses into independent segments and decodes 16 of them per
Neural Engine dispatch. A GGUF without encoder weights, such as the published
TranscribePsy Core ML distribution, loads only with its encoder sidecar; see
[Neural Engine sidecars](docs/backends.md#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars).

Nemotron 3 Diarization and Sortformer v2.1 can also run their whole network,
encoder and speaker head, on fixed-capacity diarizer sidecars named
`<model>-diarizer-<frames>.mlmodelc`. Offline, long-form, and live calls use
the smallest capacity that fits and fall back to ggml above it. On an Apple M4
they run 1.4-2.7x faster than ggml Metal. On an M3 Ultra, whose GPU is faster,
they only tie with it (0.70-1.17x), so leave them out on Max and Ultra
machines; see [the diarizer guide](docs/backends.md#core-ml-diarizer-sidecars).

## Backend selection

`parakeet --backend NAME` and `EngineOptions::backend` select the runtime device;
`parakeet-fit-params --backend NAME` and `FitOptions::backend` project memory for
the same device. Explicit requests fail if unavailable. Hexagon is experimental and requires
a staged DSP library; see [docs/backends.md](docs/backends.md) for names,
automatic placement and validation coverage, and
[docs/moss-transcribe.md](docs/moss-transcribe.md#hexagon-npu) for
MOSS-Transcribe-Diarize on the NPU.

## Performance

`RTF = inference_time / audio_duration`; lower is faster. Dated CI snapshots,
multi-machine campaigns, GPU diarization, accuracy, build pins and timing
methods live in [docs/performance.md](docs/performance.md).

## Documentation

| Topic | Where |
|---|---|
| Build modes, CMake options, repository layout | [docs/build.md](docs/build.md) |
| Backends, runtime selection, Core ML encoder sidecar | [docs/backends.md](docs/backends.md) |
| Model download and GGUF conversion | [docs/models.md](docs/models.md) |
| Public C++ API and Sortformer AOSC | [docs/api.md](docs/api.md) |
| CLI, microphone examples, memory-fit preflight | [docs/cli.md](docs/cli.md) |
| Tests and NeMo parity | [docs/testing.md](docs/testing.md) |
| MOSS-Transcribe-Diarize (conversion, CLI, API, tests) | [docs/moss-transcribe.md](docs/moss-transcribe.md) |
| Benchmark method, CI snapshots, decode optimization history | [docs/performance.md](docs/performance.md) |
| Desktop benchmark coverage (CTC, IndicConformer, TDT, Unified, EOU, Nemotron ASR, Sortformer batch and streaming, Nemotron 3 Diarization, MOSS-Transcribe-Diarize), metrics and runner limits | [Benchmark coverage](../../docs/PERFORMANCE.md#benchmark-coverage) |

## License

Code is Apache-2.0. NVIDIA Parakeet CTC, Unified RNN-T, TDT and Sortformer
weights are CC-BY-4.0. [IndicConformer](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
weights are MIT; [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
and [TranscribePsy](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub) weights
are Apache-2.0. `parakeet_realtime_eou_120m-v1` uses the
NVIDIA Open Model License. Nemotron 3.5 ASR Streaming 0.6B and Nemotron 3
Diarization use OpenMDW-1.1; redistribution of their weights must retain the
license and applicable origin notices.
No weights are shipped by this repository. [NOTICE](NOTICE) lists the
reference implementations, dependencies and model-weight sources.
