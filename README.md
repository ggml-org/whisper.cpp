# qvac-fabric-speech.cpp

On-device speech and audio AI in C++17 on [ggml](https://github.com/tetherto/qvac-ext-ggml): speech-to-text, speaker diarization, end-of-utterance detection, text-to-speech, voice cloning, sound effects, speech-to-speech, enhancement, and music generation.

| Property | Value |
|---|---|
| Build | Feature-gated CMake superbuild over Whisper and three native engines |
| Runtime | No Python, PyTorch, NeMo, or ONNX Runtime at inference time |
| Model formats | Whisper/Silero GGML `.bin`; Parakeet, TTS, and AudioGen GGUF |
| Desktop | Linux, macOS, Windows |
| Mobile | Android arm64-v8a, iOS arm64; model and backend coverage varies by engine |
| Backends | CPU, Metal, Vulkan, OpenCL, CUDA; selected models also support explicit Hexagon placement or optional Apple Core ML sidecars |
| Shared ggml | One `ggml-speech` dependency from [qvac-ext-ggml@speech](https://github.com/tetherto/qvac-ext-ggml/tree/speech) |

## Supported models

The engine guides own model formats, quantization, backend validation, and input
requirements. An available ggml backend does not imply every model has been
validated on it.

| Engine | Model families | Capabilities and limitations |
|---|---|---|
| [Whisper](docs/WHISPER.md) | tiny, base, small, medium, large v1/v2/v3, large-v3-turbo; Silero v5.1.2/v6.2.0 | ASR, multilingual-to-English translation, VAD, tinydiarize speaker turns; metadata-only memory fit from a model or its weightless description; `.en` checkpoints are English-only and turbo is intended for transcription |
| [Parakeet](engines/parakeet/README.md) | CTC, Unified RNN-T, TDT, IndicConformer, realtime EOU, Nemotron 3.5 ASR | Offline, streaming, long-form ASR and end-of-turn detection; IndicConformer CTC requires a language selection |
| Parakeet | Sortformer v1/v2/v2.1, Nemotron 3 Diarization | Speaker diarization and attributed ASR; Sortformer supports up to four speakers, Nemotron up to eight |
| Parakeet | MOSS-Transcribe-Diarize | One-pass transcription, speakers, timestamps, hotwords and metadata-only memory fit including scheduler storage and CPU scratch; CPU/Metal validated on Spanish/Chinese; explicit CUDA/Vulkan selection with generated-model CPU parity and quantized checkpoint smoke coverage; English long-form skips spans in both native and reference runs |
| [TTS](engines/tts/README.md) | Chatterbox Turbo/Multilingual, Supertonic 1/2/3, Parler mini/large/Indic, CosyVoice3, Audio8 | Synthesis with preset, described, or cloned voices, depending on the model; streaming support varies by API and CLI |
| TTS | Pocket TTS | English synthesis and incremental CPU streaming; cloning requires encoder-enabled weights |
| TTS | MOSS-TTS-v1.5 / MOSS-TTSD | Multilingual synthesis, cloning, dialogue and streaming; CPU/Metal validated; CUDA/Vulkan generated-model parity and synthesis/streaming/cancellation checks (see scoped validation in the TTS guide); numerical reference parity pending |
| TTS | MOSS-SoundEffect-v2, MOSS-Speech | Sound-effect generation and spoken replies respectively; metadata-only memory fit for SoundEffect; CPU/Metal generation validated; CUDA/Vulkan generated-model parity and quantized SoundEffect/Speech checkpoint smoke coverage |
| TTS | LavaSR denoiser/enhancer | Speech denoising and bandwidth extension |
| [AudioGen](engines/audiogen/README.md) | ACE-Step v15 turbo/sft/base | Music generation and editing; base additionally supports lego stems |
| AudioGen | MiniMax-Music3 | Desktop music generation on CPU or GPU, with a metadata-only [memory-fit preflight](engines/audiogen/docs/memory-fit.md); unavailable on Android/iOS |

CosyVoice3 uses capability-checked flash attention for single-token LM decoding
on CUDA and Metal. Supported CUDA backends reuse decode graphs within masked
128-token cache windows, allowing CUDA graph capture. Unsupported backends or
head dimensions retain their existing attention path.

CosyVoice3 flow uses F16 key/value operands with F32 accumulation for
capability-checked Vulkan flash attention, falling back to F32 operands or
regular attention when required.

CosyVoice3 expands bf16 flow weights to f32 on ARM CPUs to avoid scalar bf16
matmuls; f16 remains the recommended ARM flow bundle. Its benchmark supports
independent LM, flow and HiFT model overrides for
[stage profiling](engines/tts/docs/cosyvoice3.md#stage-profiling).

### Apple Core ML sidecars

Optional sidecars accelerate a stage while the rest of the pipeline stays on
ggml. They default to disabled and have model-specific input and fallback
contracts; see the engine guides before exporting or deploying one.

| Engine | Accelerated stage | Guide |
|---|---|---|
| Whisper | Encoder | [Whisper Core ML](third_party/whisper.cpp/README.md#core-ml-support) |
| Parakeet | Eligible ASR and Sortformer v2.1 encoder paths | [Backend and routing contracts](engines/parakeet/docs/backends.md#core-ml-encoder-sidecar) |
| Parakeet | MOSS-Transcribe-Diarize audio encoder | [MOSS-Transcribe-Diarize](engines/parakeet/docs/moss-transcribe.md#core-ml-encoder-sidecar) |
| TTS | Supertonic vocoder, Audio8 codec synthesis | [Supertonic](engines/tts/docs/supertonic.md#core-ml-vocoder-sidecar), [Audio8](engines/tts/docs/audio8.md#core-ml-codec-sidecar) |
| TTS | MOSS-TTS/TTSD codec decoder, MOSS-SoundEffect DiT and VAE, MOSS-Speech speech tokenizer | [MOSS](engines/tts/docs/moss.md#core-ml-codec-decoder), [SoundEffect](engines/tts/docs/moss.md#core-ml-dit-and-vae), [Speech](engines/tts/docs/moss.md#core-ml-speech-tokenizer) |
| AudioGen | ACE-Step VAE decoder | [AudioGen backends](engines/audiogen/docs/backends.md#core-ml-vae-decoder-sidecar) |

### Hexagon NPU (Snapdragon)

Selected models run end-to-end on the Hexagon HTP accelerator via
`--backend hexagon` (CLI) or `EngineOptions::backend = "hexagon"` (API).
Each guide names the validated quantisation tier, the runtime env flags
that recover the measured speedup, and how the backend compares to CPU
and OpenCL on the same device.

| Engine | Models | Guide |
|---|---|---|
| TTS | Supertonic 3, Parler-TTS mini, CosyVoice3, Audio8 | [Audio8 Hexagon](engines/tts/docs/audio8.md#hexagon-npu-snapdragon), and the matching sections in the other model guides |
| AudioGen | ACE-Step v15 turbo (`q8_0` DiT) | [AudioGen Hexagon](engines/audiogen/docs/backends.md#hexagon-npu) |

## Getting started

Install a shared speech-branch ggml using [docs/BUILD.md](docs/BUILD.md), then
configure from this checkout with its install prefix:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build --parallel
```

For a first transcription on a single-config build:

```sh
bash third_party/whisper.cpp/models/download-ggml-model.sh base.en
./build/bin/whisper-cli -m third_party/whisper.cpp/models/ggml-base.en.bin \
  -f third_party/whisper.cpp/samples/jfk.wav
```

The [build guide](docs/BUILD.md#windows-and-multi-config-generators) provides
Windows and multi-config commands. Use [docs/CLI.md](docs/CLI.md) to choose a
tool, and the engine's model guide to acquire its required weights.

## Performance

`RTF = inference_time / audio_duration`; lower is faster. These are dated
measurements with different workloads and timing boundaries, not universal
speed guarantees. Full methods, build pins and limitations live in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) and the linked engine reports.

| Task | Model | Recorded measurements |
|---|---|---|
| ASR | Parakeet TDT 0.6b v3 | September 2026 GPU lanes: short-clip RTF 0.0007–0.0055; [method and full table](engines/parakeet/docs/performance.md#multi-machine-benchmark-2026-09) |
| TTS | Supertonic 3 | September 2026 GPU lanes: process wall 0.61–0.82 s, including model load; [full table](engines/tts/docs/performance.md#supertonic-3-multi-machine-benchmark-2026-09) |
| TTS | Audio8 | September 2026 GPU campaign: RTF 0.20–0.65, with later RTX 5090 measurements down to 0.116; [full table and follow-up](engines/tts/docs/performance.md#audio8-multi-machine-benchmark-2026-09) |
| Music | ACE-Step 1.5 | September 2026 CUDA/Vulkan/Metal lanes: generation 1,338–9,016 ms, RTF 0.139–0.957; [full table](engines/audiogen/docs/performance.md#ace-step-15-multi-machine-benchmark-2026-09) |
| Music | MiniMax-Music3 f16 | September 2026 RTX 5090: two-minute song in 73.9 s CUDA / 84.2 s Vulkan, excluding model load; [full table](engines/audiogen/docs/performance.md#minimax-music3-f16-on-rtx-5090-2026-09) |

## Use in QVAC

These engines ship inside [QVAC](https://github.com/tetherto/qvac) as SDK
addons consuming the `speech-cpp` vcpkg port. For JavaScript/TypeScript APIs,
Bare runtime integration, and desktop/mobile applications, see QVAC.

| QVAC addon | Wraps | `speech-cpp` features |
|---|---|---|
| `@qvac/asr-ggml` | ASR, diarization, end-of-utterance | `whisper`, `parakeet` |
| `@qvac/tts-ggml` | Synthesis, voice cloning, enhancement | `tts` |
| `@qvac/audiogen-ggml` | Music generation | `audiogen` |
| `@qvac/bci-whispercpp` | Brain-computer interface transcription | `whisper` |

## Licenses

Code and model weights have separate terms. No model weights are shipped in
this source tree. Engine license sections and `NOTICE` files identify sources
and model-specific exceptions.

| Component | Code | Model weights |
|---|---|---|
| [Whisper](third_party/whisper.cpp/LICENSE) | MIT | OpenAI Whisper MIT; Silero under its model terms |
| [Parakeet](engines/parakeet/README.md#license) | Apache-2.0 | NVIDIA Parakeet/Sortformer CC-BY-4.0; EOU NVIDIA Open Model License; Nemotron OpenMDW-1.1; IndicConformer MIT; MOSS-Transcribe-Diarize Apache-2.0 |
| [TTS](engines/tts/NOTICE) | MIT | Chatterbox MIT; Pocket CC-BY-4.0; Supertonic OpenRAIL-M; Parler, CosyVoice3, Audio8, LavaSR and MOSS models Apache-2.0 |
| [AudioGen](engines/audiogen/NOTICE) | MIT | ACE-Step 1.5 MIT; Qwen3-Embedding Apache-2.0; MiniMax-Music3 Community License |

## Documentation

| Topic | Where |
|---|---|
| Architecture and repository layout | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Building and consuming packages | [docs/BUILD.md](docs/BUILD.md) |
| CLI selection and examples | [docs/CLI.md](docs/CLI.md) |
| Performance overview and reports | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| Whisper models and formats | [docs/WHISPER.md](docs/WHISPER.md) |
| Whisper subtree deltas and synchronization | [PATCHES.md](third_party/whisper.cpp/PATCHES.md), [docs/UPSTREAM-SYNC.md](docs/UPSTREAM-SYNC.md) |
| ASR, diarization, end-of-utterance | [Parakeet](engines/parakeet/README.md) |
| Synthesis, voice cloning, sound effects, speech-to-speech, enhancement | [TTS](engines/tts/README.md) |
| Music generation and editing | [AudioGen](engines/audiogen/README.md) |
| Benchmark quality diagnostics and model preparation | [Music alignment guide](scripts/benchmarks/music-alignment.md) |
| Historical development and integration reports | [Chatterbox](engines/tts/docs/history/chatterbox-port.md), [Supertonic](engines/tts/docs/history/supertonic-port.md), [Pocket](engines/tts/docs/history/pocket-integration.md), [Parakeet](engines/parakeet/docs/history/parakeet-port.md), [Strix Halo](engines/audiogen/docs/history/strix-halo.md) |
