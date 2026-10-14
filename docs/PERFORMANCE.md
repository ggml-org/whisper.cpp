# Performance across modalities and hosts

Part of the [qvac-fabric-speech.cpp documentation](../README.md).

These dated snapshots preserve their original workloads and build versions.
For full campaigns and reproduction methods, see [Parakeet](../engines/parakeet/docs/performance.md),
[TTS](../engines/tts/docs/performance.md), and [AudioGen](../engines/audiogen/docs/performance.md).
Quality diagnostics and MiniMax benchmark preparation are documented in the
[music alignment guide](../scripts/benchmarks/music-alignment.md).

## Benchmark coverage

The desktop benchmark workflow (`speech-benchmark-desktop.yml`, driven by
`scripts/benchmarks/families.json`) measures each family below on the hosted
Linux (CPU) and self-hosted macOS (Metal) runners and scores a correctness
metric where the harness has one; `none` marks performance-only rows. The
last column records what keeps a row out of the default `all` dispatch, off
one runner, or short of the model line's full size range.

| Family | Model | Workload | Correctness | Limit |
|---|---|---|---|---|
| `whisper` | tiny, base, small | `jfk.wav` transcription | WER | larger sizes excluded to save runner minutes |
| `parakeet` | TDT 0.6b v3 f16 | `jfk.wav`, Core ML vs Metal on macOS | WER | |
| `parakeet-unified` | Unified RNN-T 0.6b q8_0 | `jfk.wav`, Core ML vs Metal on macOS | WER | |
| `parakeet-eou` | realtime EOU 120m f16 | `jfk.wav`, Core ML vs Metal on macOS | WER | |
| `parakeet-nemotron` | Nemotron 3.5 ASR 0.6b f16 | `jfk.wav`, Core ML vs Metal on macOS | WER | |
| `parakeet-ctc` | CTC 0.6b q8_0 | `jfk.wav` | WER | |
| `parakeet-indic` | IndicConformer CTC q8_0 | `hi-16k.wav`, `--language hi` | WER, `none` normalizer | |
| `sortformer` | Sortformer v1 q8_0 | `abcba.wav` batch diarization | DER | |
| `sortformer-streaming` | streaming Sortformer v2.1 q8_0 | `abcba.wav` streaming session | DER | |
| `nemotron-diarization` | Nemotron 3 Diarization q8_0 (official GGUF) | `abcba.wav` long-form diarization | DER | |
| `moss-transcribe` | MOSS-Transcribe-Diarize q8_0 | `jfk.wav` | WER of the segment text | |
| `vad`, `vad-v6` | Silero v5.1.2, v6.2.0 | `jfk.wav` speech segments | F1 | |
| `chatterbox` | Chatterbox Turbo q8_0 | shared English prompt | TTS WER | |
| `chatterbox-multilingual` | Chatterbox Multilingual q8_0 | shared English prompt, `--language en` | TTS WER | |
| `supertonic1`, `supertonic2`, `supertonic` | Supertonic 1, 2, 3 q8_0 | shared English prompt | TTS WER | |
| `parler`, `parler-indic` | Parler mini v1, Indic Parler q8_0 | shared English prompt, sampled | TTS WER | |
| `cosyvoice` | Fun-CosyVoice3-0.5B | shared English prompt | TTS WER | |
| `audio8-tts` | Audio8-TTS-Preview-0.6B q8_0 | shared English prompt | TTS WER | |
| `audio8` | Audio8 codec q8_0 | `jfk.wav` reconstruction | SI-SDR, STOI | |
| `lavasr` | LavaSR denoiser and enhancer f16 | noisy `jfk.wav` enhancement | SI-SDR, STOI | |
| `moss-tts` | MOSS-TTS-v1.5 f16 | shared English prompt | TTS WER | macOS only, opt-in: 17 GB backbone |
| `moss-sfx` | MOSS-SoundEffect-v2 q8_0 | 5 s sound effect, 25 steps | none | macOS only, opt-in: CPU diffusion too slow |
| `moss-speech` | MOSS-Speech q8_0 | spoken reply to `jfk.wav` | none | macOS only, opt-in: 11 GB of weights |
| `acestep` | ACE-Step v15 turbo q8_0 | 4 s generation | CLAP, opt-in | |
| `minimax` | MiniMax-Music3 q4_k_m | opt-in CLAP diagnostics only | CLAP, opt-in | no performance-only invocation |

Supported models without a benchmark family, and why:

- Whisper medium, large v1/v2/v3, large-v3-turbo and tinydiarize: left out on purpose to keep runner minutes down; `whisper_sizes` accepts tiny, base and small.
- Parakeet CTC 1.1b and TDT 1.1b, streaming Sortformer v2, Parler-TTS large and Pocket TTS: no GGUF in the model registry to pin.
- MOSS-TTSD: its input is a two-speaker dialogue with reference audio and transcripts, which the shared single-speaker prompt and intelligibility scorer do not model.
- ACE-Step sft and base: only the turbo checkpoint is registered under the benchmark's prefix.
- MOSS-SoundEffect and MOSS-Speech outputs are timed but not scored: no sound-effect or spoken-reply reference exists in the harness.

## ASR, end-of-utterance, diarization

CI numbers from the published `@qvac/asr-ggml@0.1.1` addon ([run 31603189415](https://github.com/tetherto/qvac/actions/runs/31603189415), 2026-08-12), `q8_0` GGUFs, 1 warmup plus 5 timed runs, host `qvac-ubuntu2204-x64-gpu` (CPU: Intel Core i5-13500, GPU: NVIDIA RTX 4000 SFF Ada, Vulkan). Full table: [engines/parakeet/README.md](../engines/parakeet/docs/performance.md#asr-addon-ci-2026-08-12).

| Model | CPU RTF | CPU wall | Vulkan RTF | Vulkan wall |
|---|--:|--:|--:|--:|
| Parakeet CTC | 0.112 | 2256 ms | 0.0022 | 43 ms |
| Parakeet TDT | 0.130 | 2607 ms | 0.0044 | 88 ms |
| Parakeet EOU | 0.051 | 1034 ms | 0.0034 | 68 ms |
| Sortformer | 0.046 | 922 ms | 0.0019 | 38 ms |
| Sortformer streaming | 0.032 | 646 ms | 0.0034 | 69 ms |
| Whisper base | 0.035 | 699 ms | 0.0057 | 117 ms |
| Whisper small | 0.122 | 2453 ms | 0.0098 | 200 ms |

### speech-cpp CI (2026-09-07, CPU + macOS)

CPU only on Linux; macOS whisper rows run on Metal (`MTL0`).

| Model | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Parakeet CTC 0.6b q8_0 | linux | ggml-cpu | 2111 | 0.192 | 858 |
| Parakeet CTC 0.6b q8_0 | macos | ggml-cpu | 184 | 0.0170 | 914 |
| Whisper tiny | linux | (CPU) | 1035 | 0.0940 | 182 |
| Whisper tiny | macos | Metal | 565 | 0.0510 | 240 |
| Whisper base | linux | (CPU) | 1906 | 0.173 | 298 |
| Whisper base | macos | Metal | 584 | 0.0530 | 360 |
| Whisper small | linux | (CPU) | 6122 | 0.557 | 797 |
| Whisper small | macos | Metal | 564 | 0.0510 | 878 |

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

## Text-to-speech

CI numbers from the published `@qvac/tts-ggml@0.6.2` addon ([run 31603192731](https://github.com/tetherto/qvac/actions/runs/31603192731), 2026-08-12), `q4_0` GGUFs, same host. Full table: [engines/tts/README.md](../engines/tts/docs/performance.md#ci-benchmarks-2026-08-12-linux-x86-64).

| Model | CPU RTF | Vulkan RTF | Vulkan wall | Vulkan tok/s |
|---|--:|--:|--:|--:|
| Chatterbox Turbo | 1.54 | 0.099 | 410 ms | 173 |
| Chatterbox Multilingual | 5.81 | 0.182 | 1036 ms | 77 |
| Supertonic | 0.113 | 0.018 | 78 ms | 952 |
| Supertonic Multilingual | 0.101 | 0.013 | 84 ms | 1087 |
| Supertonic 3 | 0.225 | 0.029 | 118 ms | 631 |

### speech-cpp CI (2026-09-07, CPU + macOS)

| Engine | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Chatterbox: chatterbox-t3-turbo-q8_0 | linux | CPU | 8078 | — | 1835 |
| Chatterbox: chatterbox-t3-turbo-q8_0 | macos | CPU | 2626 | — | 1746 |
| Supertonic: supertonic3-q8_0 | linux | (CPU) | 708 | 0.484 | 876 |
| Supertonic: supertonic3-q8_0 | macos | (CPU) | 49 | 0.0340 | 584 |
| Parler: parler-mini-v1-q8_0 | linux | CPU | 249362 | 18.8 | 1964 |
| Parler: parler-mini-v1-q8_0 | macos | MTL0 (Metal) | 12204 | 0.582 | 1307 |
| Cosyvoice: cosyvoice3-llm-q8_0 | linux | CPU | 67695 | 18.0 | 1416 |
| Cosyvoice: cosyvoice3-llm-q8_0 | macos | CPU | 46825 | 12.3 | 1467 |

`—` means RTF was not reported in the benchmark artifact.

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

## Music generation & other engines (speech-cpp CI, 2026-09-07)

| Engine | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Acestep: Qwen3-Embedding-0.6B-Q8_0 | linux | (CPU) | 36817 | 9.20 | 1702 |
| Acestep: Qwen3-Embedding-0.6B-Q8_0 | macos | (CPU) | 6698 | 1.67 | 2222 |
| Lavasr: lavasr-denoiser-f16 | linux | (CPU) | 4102 | 0.684 | 155 |
| Lavasr: lavasr-denoiser-f16 | macos | (CPU) | 2090 | 0.348 | 209 |

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

## Apple silicon

| Model | Host | Backend | Quantization | RTF | vs real-time |
|---|---|---|---|--:|--:|
| Parakeet TDT 0.6b v3 | Mac mini M4 (CI, `mac-mini-m4-gpu`) | Metal | `q8_0` | 0.015 | 67x |
| Parakeet CTC | Mac mini M4 (CI, `mac-mini-m4-gpu`) | Metal | `q8_0` | 0.011 | 88x |
| Whisper small | Mac mini M4 (CI, `mac-mini-m4-gpu`) | Metal | `q8_0` | 0.027 | 37x |
| Chatterbox Turbo | Mac Studio M3 Ultra | Metal | `q4_0` | 0.16 | 6.4x |
| Chatterbox Turbo | Mac Studio M3 Ultra | CPU (NEON) | `q4_0` | 1.05 | 0.96x |
| Chatterbox Multilingual (`--cfm-steps 7`) | Mac Studio M3 Ultra | Metal | `q4_0` | 0.30 | 3.3x |
| Chatterbox Multilingual | Apple M4 | Metal | `q4_0` | 1.37 | 0.73x |

## Streaming latency

Chatterbox on Apple M4 Metal, 317 speech tokens (12.7 s of audio), `--stream-first-chunk-tokens 10 --stream-chunk-tokens 25 --stream-cfm-steps 1`. Full table: [TTS streaming performance](../engines/tts/docs/performance.md#streaming-mode--low-latency-playback).

| Metric | Value |
|---|--:|
| first audio out | 279 ms |
| steady-state chunk RTF | 0.30 to 0.63 |
| overall RTF | 0.90 |

On-device Android and iOS performance is tracked by the benchmark lanes in [QVAC](https://github.com/tetherto/qvac).

## Brain-computer interface (2026-08-12)


CI numbers from the published `@qvac/bci-whispercpp@0.6.0` addon ([run 31602627344](https://github.com/tetherto/qvac/actions/runs/31602627344), 2026-08-12), `ggml-bci-windowed` model. Throughput in tokens/s, higher is better.

| Host | CPU tok/s | Vulkan tok/s | Vulkan wall |
|---|--:|--:|--:|
| Linux x86-64 (i5-13500 / RTX 4000 SFF Ada) | 27.0 | 355.6 | 42 ms |
| Windows x64 (`qvac-win25-x64-gpu`) | 20.1 | 36.0 | 349 ms |
| Linux arm64 (`ubuntu-24.04-arm`, CPU-only lane) | 16.8 | n/a | n/a |

The macOS arm64 lane runs on the GitHub-hosted `macos-26` runner, whose virtualised Metal device is not representative (6.6 tok/s vs 398 tok/s on the previously used self-hosted M-series box), so it is omitted here.
