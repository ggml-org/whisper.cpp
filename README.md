# qvac-fabric-speech.cpp

On-device speech and audio AI in C++17 on [ggml](https://github.com/tetherto/qvac-ext-ggml): speech-to-text, speaker diarization, end-of-utterance detection, text-to-speech, voice cloning, sound effects, speech-to-speech, enhancement, and music generation.

| Property | Value |
|---|---|
| Build | Feature-gated CMake superbuild over Whisper and three native engines |
| Runtime | No Python, PyTorch, NeMo, or ONNX Runtime at inference time |
| Model formats | Whisper/Silero GGML `.bin`; Parakeet, TTS, and AudioGen GGUF |
| Desktop | Linux, macOS, Windows |
| Mobile | Android arm64-v8a, iOS arm64; model and backend coverage varies by engine |
| Backends | CPU, Metal, Vulkan, OpenCL, CUDA, WebGPU (Dawn, opt-in; see [WebGPU](#webgpu-dawn)); selected models also support explicit Hexagon placement or optional Apple Core ML sidecars |
| Shared ggml | One `ggml-speech` dependency from [qvac-ext-ggml@speech](https://github.com/tetherto/qvac-ext-ggml/tree/speech) |

## Supported models

The engine guides own model formats, quantization, backend validation, and input
requirements. An available ggml backend does not imply every model has been
validated on it.

| Engine | Model families | Capabilities and limitations |
|---|---|---|
| [Whisper](docs/WHISPER.md) | tiny, base, small, medium, large v1/v2/v3, large-v3-turbo; Silero v5.1.2/v6.2.0 | ASR, multilingual-to-English translation, VAD, tinydiarize speaker turns; metadata-only memory fit from a model or its weightless description; `.en` checkpoints are English-only and turbo is intended for transcription; tiny/base/small validated on WebGPU |
| [Parakeet](engines/parakeet/README.md) | CTC, Unified RNN-T, TDT, TranscribePsy, IndicConformer, realtime EOU, Nemotron 3.5 ASR | Offline, streaming, long-form ASR and end-of-turn detection; IndicConformer CTC requires a language selection; TranscribePsy ships as ready-made GGUF and Core ML files for eight European languages and runs on the Hexagon NPU |
| Parakeet | Sortformer v1/v2/v2.1, Nemotron 3 Diarization | Speaker diarization and attributed ASR; Sortformer supports up to four speakers, Nemotron up to eight |
| Parakeet | MOSS-Transcribe-Diarize | One-pass transcription, speakers, timestamps, hotwords and metadata-only memory fit including scheduler storage and CPU scratch; fused CPU encoder attention; CPU/Metal validated on Spanish/Chinese; Metal decode reuses one graph per cache window (29-42% faster decode on M3 Ultra, identical transcripts); explicit CUDA/Vulkan/OpenCL/Metal/Hexagon selection with generated-model CPU parity and quantized checkpoint smoke coverage; English long-form skips spans in both native and reference runs |
| [TTS](engines/tts/README.md) | Chatterbox Turbo/Multilingual, Supertonic 1/2/3, Parler mini/large/Indic, CosyVoice3, Audio8 | Synthesis with preset, described, or cloned voices, depending on the model; streaming support varies by API and CLI; Supertonic 3 also validated on WebGPU |
| TTS | Pocket TTS | English synthesis and incremental CPU/Metal/CUDA/Vulkan/OpenCL streaming; cloning requires encoder-enabled weights |
| TTS | MOSS-TTS-v1.5 / MOSS-TTSD | Multilingual synthesis, cloning, dialogue and streaming; single-reference dialogue avoids duplicate reference encoding; selective text-control projection on Metal; CPU/Metal validated; CUDA/Vulkan generated-model parity and synthesis/streaming/cancellation checks (see scoped validation in the TTS guide); explicit Hexagon NPU placement with a `q4_0` transformer; numerical reference parity pending |
| TTS | MOSS-SoundEffect-v2, MOSS-Speech | Sound-effect generation and spoken replies respectively; metadata-only memory fit for SoundEffect; CPU/Metal generation validated; CUDA/Vulkan generated-model parity and quantized SoundEffect/Speech checkpoint smoke coverage; explicit Hexagon NPU placement for SoundEffect, with reference parity, and for the Speech LM and speech tokenizer |
| TTS | LavaSR denoiser/enhancer | Speech denoising and bandwidth extension; GPU selection includes ARM Mali for both stages and excludes WebGPU, whose kernels lack an op the graphs need |
| [AudioGen](engines/audiogen/README.md) | ACE-Step v15 turbo/sft/base | Music generation and editing; base additionally supports lego stems |
| AudioGen | MiniMax-Music3 | Desktop music generation on CPU or GPU, optional [Core ML conditioner projection](engines/audiogen/docs/backends.md#minimax-core-ml-conditioner), and metadata-only [memory-fit preflight](engines/audiogen/docs/memory-fit.md); unavailable on Android/iOS |

MOSS cloning matches upstream reference loudness preprocessing and uses the
TTSD-specific dialogue prompt. The [TTSD example](engines/tts/docs/moss.md)
sets its model-specific sampling parameters explicitly; check generated speech
for complete dialogue rather than relying on a successful exit alone.
An optional [validated cloning runner](engines/tts/docs/moss.md#validated-cloning)
checks English dialogue with local ASR and retries rejected generations before
publishing an output WAV. ASR results that reach the decoding token limit are rejected.

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

Nemotron 3 Diarization preparation supports additional weight precisions via
`convert_nemotron_diarization.py --outtype`. CPU/CUDA offline validation covers
F16, BF16, Q6_K, Q5_K, Q4_K, and NVFP4. Q4_K/Q6_K also pass offline
Metal and Vulkan validation; NVFP4 runs on Vulkan but falls back to CPU
matmuls on Metal. Streaming quality varies by precision and backend.
See the [quantization measurements](engines/parakeet/docs/performance.md#nemotron-3-diarization-quantizations)
before selecting a format. MXFP4 failed CPU quality checks.

### Experimental Parakeet-Mosaic

[Mosaic](engines/parakeet/docs/mosaic.md) uses a frozen mixed-precision
recipe with ordinary GGML kernels. The shared 486.7 MB Mosaic and 339.6 MB
[Mosaic Compact](engines/parakeet/docs/mosaic.md#shared-mosaic-and-mosaic-compact-models)
also support separate, matching optional Core ML encoder sidecars. The earlier experimental
[`parakeet-affine-v1` format](engines/parakeet/docs/backends.md#experimental-packed-affine-tdt-recipe)
uses integer carriers, with reconstruction by default and opt-in Metal fusion
via `PARAKEET_AFFINE_FUSED=1`. Fusion requires the matching
[GGML dependency](engines/parakeet/docs/fused-affine-metal.md#integration-and-pr-handoff);
older dependencies and unsupported fused backends fail explicitly. Offline TDT
is the evaluated scope; packed-affine substage profiling is rejected. Recipe
receipts and measured limitations are documented in the linked guides.

Fixed TDT Core ML exports now require both `mel` and `mel_valid_mask`, and need
the updated runtime. Legacy single-input sidecars still load but must be
re-exported for padding invariance. See the
[sidecar compatibility contract](engines/parakeet/docs/backends.md#core-ml-encoder-sidecar).

### Apple Core ML sidecars

Optional sidecars accelerate a stage while the rest of the pipeline stays on
ggml. They default to disabled and have model-specific input and fallback
contracts; see the engine guides before exporting or deploying one.

| Engine | Accelerated stage | Guide |
|---|---|---|
| Whisper | Encoder | [Whisper Core ML](third_party/whisper.cpp/README.md#core-ml-support) |
| Parakeet | Eligible ASR and Sortformer v2.1 encoder paths; TDT models also run a fixed-window Neural Engine encoder and a batched TDT decoder | [Backend and routing contracts](engines/parakeet/docs/backends.md#core-ml-encoder-sidecar), [Neural Engine sidecars](engines/parakeet/docs/backends.md#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars) |
| Parakeet | Nemotron 3 Diarization and Sortformer v2.1 whole networks | [Diarizer sidecars](engines/parakeet/docs/backends.md#core-ml-diarizer-sidecars) |
| Parakeet | MOSS-Transcribe-Diarize audio encoder | [MOSS-Transcribe-Diarize](engines/parakeet/docs/moss-transcribe.md#core-ml-encoder-sidecar) |
| TTS | CosyVoice3 speech LM, flow and HiFT neural stages | [CosyVoice3 Core ML](engines/tts/docs/cosyvoice3.md#core-ml-sidecars) |
| TTS | Supertonic vocoder, Audio8 codec synthesis | [Supertonic](engines/tts/docs/supertonic.md#core-ml-vocoder-sidecar), [Audio8](engines/tts/docs/audio8.md#core-ml-codec-sidecar) |
| TTS | MOSS-TTS/TTSD codec decoder, MOSS-SoundEffect DiT and VAE, MOSS-Speech speech tokenizer | [MOSS](engines/tts/docs/moss.md#core-ml-codec-decoder), [SoundEffect](engines/tts/docs/moss.md#core-ml-dit-and-vae), [Speech](engines/tts/docs/moss.md#core-ml-speech-tokenizer) |
| AudioGen | ACE-Step VAE decoder | [AudioGen backends](engines/audiogen/docs/backends.md#core-ml-vae-decoder-sidecar) |
| AudioGen | MiniMax conditioner projection, or explicit full-network Core ML (`coreml_profile_dir` + `coreml_compute_units`, macOS 15+) | [MiniMax Core ML](engines/audiogen/docs/backends.md#minimax-core-ml-conditioner) |

MiniMax's full Core ML route is also exposed through its public C++ engine.
Build with `AUDIOGEN_MINIMAX_COREML=ON` and supply both `coreml_profile_dir` and
`coreml_compute_units`; the separate flag does not enable automatic ACE-Step
sidecars. It supports `CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE`, and `ALL` on
macOS 15+. See the [full Core ML guide](engines/audiogen/docs/minimax-coreml-validation.md)
for the matching F16 compiled profile, measurements, and remaining numerical
validation.

CosyVoice3 exports HiFT in FP32 to preserve waveform parity. Its memory-fit
preflight returns `Error` with `coreml-memory-estimate-unavailable` when
enabled sidecars are present, rather than undercounting their resident memory.
Re-export CosyVoice3 LM sidecars with the current exporter to preserve cache
feedback on macOS 15 and 26; see the [Core ML guide](engines/tts/docs/cosyvoice3.md#core-ml-sidecars).

### Hexagon NPU (Snapdragon)

Selected models run on the Hexagon HTP accelerator via `--backend hexagon`
(CLI) or the engine's `backend = "hexagon"` option (API), end to end except
for MOSS-Speech, whose reply decoder stays on the CPU or the GPU.
Each guide names the validated quantisation tier, the runtime env flags
that recover the measured speedup, and how the backend compares to CPU
and OpenCL on the same device.

| Engine | Models | Guide |
|---|---|---|
| ASR | Parakeet CTC and TDT 0.6B (`q8_0`), TranscribePsy-250M-EU-sub (`q8_0`), MOSS-Transcribe-Diarize (`q8_0`, `f16`) | [Parakeet Hexagon](engines/parakeet/docs/backends.md#hexagon-npu), [MOSS Hexagon](engines/parakeet/docs/moss-transcribe.md#hexagon-npu) |
| TTS | Chatterbox Turbo and Multilingual (`q4_0`, `q8_0`, `f16` or `f32`), Supertonic 3, Parler-TTS mini, CosyVoice3, Audio8, MOSS-TTS-v1.5 / MOSS-TTSD (`q4_0` transformer), MOSS-SoundEffect-v2 (`f16` or `q8_0`), MOSS-Speech LM (`q4_0` transformer) and speech tokenizer (`f16` codec) | [Chatterbox Hexagon](engines/tts/docs/chatterbox.md#snapdragon-hexagon-npu), [Audio8 Hexagon](engines/tts/docs/audio8.md#hexagon-npu-snapdragon), [MOSS Hexagon](engines/tts/docs/moss.md#hexagon-npu), [MOSS-SoundEffect Hexagon](engines/tts/docs/moss.md#hexagon-npu-generation), [MOSS-Speech Hexagon](engines/tts/docs/moss.md#hexagon-npu-speech-to-speech), and the matching sections in the other model guides |
| AudioGen | ACE-Step v15 turbo (`q8_0` DiT; K-quant GGUFs are refused at load) | [AudioGen Hexagon](engines/audiogen/docs/backends.md#hexagon-npu) |

### WebGPU (Dawn)

`GGML_WEBGPU=ON` (the `webgpu` feature of the `speech-cpp` / `ggml-speech`
ports) adds ggml's WebGPU backend on Google Dawn, driven natively inside the
process; no browser is involved. It registers as `WebGPU`, ranks with Vulkan
and Metal in every engine's automatic GPU selection (CUDA still wins on the
same NVIDIA card), and is selected explicitly with `--backend webgpu` or
`backend = "webgpu"`. Build and host requirements are in
[docs/BUILD.md](docs/BUILD.md#webgpu-dawn). Validation on an RTX 5090
(Dawn on Vulkan, 2026-10-20), each model against the same CPU run:

| Engine | Model | WebGPU result |
|---|---|---|
| Whisper | tiny.en, base, small | Validated: same transcripts; encoder 8x (base) to 33x (small) faster than CPU, a 2-minute clip 3.1x faster end to end |
| Parakeet | TranscribePsy 250M CTC | Validated: identical transcript, 2.8x faster end to end |
| TTS | Supertonic 3 | Validated: waveform correlation 0.997 with the CPU render, identical transcript |
| TTS | Parler-TTS mini | Not supported: the decoder returns NaN logits, which the sampler now reports as an error instead of crashing |
| TTS | CosyVoice3, Audio8 | Not supported: synthesis completes but the audio is silence or noise |
| TTS | Pocket, Chatterbox, MOSS, LavaSR | Not tested |
| AudioGen | ACE-Step, MiniMax-Music3 | Not tested; the ACE-Step LM and detokenizer keep their CPU placement on an unmeasured backend |

The failing TTS models share one cause: ggml's WGSL matmul kernels convert
activations to f16, and `test-backend-ops` on WebGPU fails exactly the
fork's large-activation (`b_absmax=1e5`) and `GGML_PREC_F32` matmul cases
that the CUDA backend needed f32 paths for. Until the WebGPU kernels gain
the same paths, those engines keep WebGPU out of their automatic selection;
an explicit request still runs so the status can be re-checked.

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

`audio8-codec-bench` can decode captured codes on CPU, Hexagon or OpenCL and
export WAV/raw F32 PCM while measuring the codec separately from generation.
See the [fixed-code decoder benchmark](engines/tts/docs/audio8.md#fixed-code-decoder-benchmark).

`RTF = inference_time / audio_duration`; lower is faster. These are dated
measurements with different workloads and timing boundaries, not universal
speed guarantees. Full methods, build pins and limitations live in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) and the linked engine reports;
[benchmark coverage](docs/PERFORMANCE.md#benchmark-coverage) lists which
model each desktop benchmark family measures and scores.

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
| [Parakeet](engines/parakeet/README.md#license) | Apache-2.0 | NVIDIA Parakeet/Sortformer CC-BY-4.0; EOU NVIDIA Open Model License; Nemotron OpenMDW-1.1; IndicConformer MIT; MOSS-Transcribe-Diarize and TranscribePsy Apache-2.0 |
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
| Native Whisper Snapdragon build and validation | [docs/WHISPER-HEXAGON.md](docs/WHISPER-HEXAGON.md) |
| Whisper subtree deltas and synchronization | [PATCHES.md](third_party/whisper.cpp/PATCHES.md), [docs/UPSTREAM-SYNC.md](docs/UPSTREAM-SYNC.md) |
| ASR, diarization, end-of-utterance | [Parakeet](engines/parakeet/README.md) |
| Synthesis, voice cloning, sound effects, speech-to-speech, enhancement | [TTS](engines/tts/README.md) |
| Music generation and editing | [AudioGen](engines/audiogen/README.md) |
| Benchmark quality diagnostics and model preparation | [Music alignment guide](scripts/benchmarks/music-alignment.md) |
| Historical development and integration reports | [Chatterbox](engines/tts/docs/history/chatterbox-port.md), [Supertonic](engines/tts/docs/history/supertonic-port.md), [Pocket](engines/tts/docs/history/pocket-integration.md), [Parakeet](engines/parakeet/docs/history/parakeet-port.md), [Strix Halo](engines/audiogen/docs/history/strix-halo.md) |

Audio8 exposes incremental PCM through `Engine::synthesize_stream()`, with codec-window
delivery on ggml and Core ML. See the [streaming API and measurement guide](engines/tts/docs/audio8.md#incremental-pcm-streaming).
