# tts-cpp

Native C++17/ggml speech synthesis, voice cloning, sound-effect generation,
and post-synthesis enhancement for the QVAC speech stack. There is no Python, PyTorch, or ONNX
Runtime dependency after models have been converted to GGUF.

This package exposes seven public synthesis engine APIs covering eight synthesis
families and ten model lines: Chatterbox Turbo, Chatterbox Multilingual,
Supertonic 1, 2, and 3, Parler-TTS, CosyVoice3, Audio8, Pocket TTS, and MOSS
Delay. MOSS-SoundEffect adds a separate text-to-sound-effects API
(`moss/sound_effect.h`), MOSS-Speech a speech-to-speech API
(`moss/speech.h`), and LavaSR is a speech-enhancement pipeline; none of the
three is a TTS synthesizer. The API
count follows the installed public headers under `include/tts-cpp/`; the two
Chatterbox families share one engine API, while the Supertonic generations
share another.

Pocket TTS runs on CPU; cloning requires encoder-enabled weights. MOSS Delay
and MOSS-TTSD are validated end to end on CPU/Metal, with numerical reference
parity pending. MOSS-SoundEffect and MOSS-Speech have CPU/Metal stage validation;
CUDA and Vulkan add generated-model CPU parity and full-checkpoint smoke coverage.
TTS streaming, cancellation/reuse and ASR intelligibility gates are described
in [MOSS GPU validation](docs/moss.md#gpu-validation-cuda-and-vulkan); smoke coverage does
not establish general speech quality. OpenCL remains untested here. See the [API matrix](docs/api.md) for
streaming support and the [model guide](docs/models.md) for storage tiers.

This in-tree package consumes the shared `ggml-speech` dependency. Build and
vcpkg consumption instructions are in [docs/build.md](docs/build.md).

## Capabilities

### Supported models and backends

Backends below are the paths validated or explicitly implemented by each
engine, not every backend ggml can compile.

| Model line | Languages | Voice source | Native rate | CPU | Metal | Vulkan | OpenCL | CUDA |
|---|---|---|---:|:---:|:---:|:---:|:---:|:---:|
| Chatterbox Turbo | English | built-in or zero-shot reference WAV/profile | 24 kHz | yes | yes | yes | yes | yes |
| Chatterbox Multilingual | 23 | built-in or zero-shot reference WAV/profile | 24 kHz | yes | yes | yes | yes | yes |
| Supertonic 1 | English | preset or external style tensors/JSON | 44.1 kHz | yes | yes (+ optional Core ML sidecar) | yes | yes | yes |
| Supertonic 2 | `en`, `ko`, `es`, `pt`, `fr` | preset or external style tensors/JSON | 44.1 kHz | yes | yes (+ optional Core ML sidecar) | yes | yes | yes |
| Supertonic 3 | 31 languages plus `na` | preset or external style tensors/JSON | 44.1 kHz | yes | yes (+ optional Core ML sidecar) | yes | yes | yes |
| Parler-TTS mini/large/Indic | English or 21 Indic languages | natural-language description | 44.1 kHz | yes | yes | yes | yes | yes |
| Fun-CosyVoice3-0.5B | model-advertised multilingual text | baked voice or zero-shot/cross-lingual reference WAV; instruct controls | 24 kHz | yes | yes | yes | yes | yes |
| Audio8-TTS-Preview-0.6B | multilingual checkpoint vocabulary | model voice or zero-shot reference WAV + transcript | 44.1 kHz | yes | yes (+ optional Core ML sidecar) | yes | yes | yes |
| Pocket TTS | English | prepared voice; cloning requires encoder-enabled weights | 24 kHz | yes | no | no | no | no |
| MOSS Delay (MOSS-TTS-v1.5 / MOSS-TTSD) | model-advertised multilingual text | model voice, zero-shot reference WAV, or two-speaker dialogue references; pause/duration/pronunciation controls | 24 kHz | yes | yes (+ optional Core ML sidecar) | scoped¹ | untested | scoped¹ |
| MOSS-SoundEffect-v2 (text to sound effects) | text prompt | none | 48 kHz | yes | yes (+ optional Core ML sidecars) | scoped¹ | untested | scoped¹ |
| MOSS-Speech (speech to speech) | spoken English and Chinese | built-in default voice or reference WAV | 24 kHz | yes | yes (+ optional Core ML sidecar) | scoped¹ | untested | scoped¹ |
| LavaSR denoiser | language agnostic | input PCM | rate preserving | yes | yes | yes | yes | yes |
| LavaSR enhancer | language agnostic | input PCM | 48 kHz | yes | yes | yes | yes | yes |

¹ MOSS CUDA/Vulkan: generated-model CPU parity; quantized TTS/TTSD,
SoundEffect and Speech checkpoint smoke cases. Vulkan also covers the
F16 TTS/TTSD/SoundEffect and BF16 Speech cases. TTS quality gates retain
known upstream limitations; see the linked MOSS guide for details.

LavaSR denoiser and enhancer GPU selection includes ARM Mali/Immortalis
through Vulkan when GPU execution is requested.

Supertonic 3, Parler-TTS mini, CosyVoice3 and Audio8 also support explicit
Hexagon placement on Snapdragon; see their model guides for setup and validated
tiers. Audio8's corrected baseline uses `q8_0` quantisation with
`--greedy` sampling under `--backend hexagon`; see
[Audio8](docs/audio8.md#hexagon-npu-snapdragon) for the full-compute tuning
recipe (`OPPOLL=1`, `OPSTAGE=3`, `OPFUSION=1`), the corrected single-prompt
baseline, and the remaining cross-backend correctness limits. Optional
Apple Core ML sidecars accelerate the Supertonic vocoder, the Audio8 codec, and
the MOSS-TTS codec decoder, MOSS-SoundEffect DiT and VAE, and MOSS-Speech
speech tokenizer. Export, routing, fallback, and per-call status are documented in
[Supertonic](docs/supertonic.md#core-ml-vocoder-sidecar),
[Audio8](docs/audio8.md#core-ml-codec-sidecar), and
[MOSS](docs/backends.md#apple-core-ml-sidecars).

Chatterbox Multilingual has native tokenization for 18 languages; Japanese,
Hebrew, Russian, Chinese and Hindi need external preprocessing. See
[CLI setup](docs/cli.md#run--end-to-end-text--wav) for the supported paths.

Backend selection and model-specific validation are documented in
[docs/backends.md](docs/backends.md) and [docs/testing.md](docs/testing.md).
Metadata-only memory preflight covers MOSS Delay/TTSD and MOSS-SoundEffect,
including text/diffusion/decode workloads for sound effects. Entry points are indexed in
[docs/api.md](docs/api.md#memory-fit-preflight).

## Performance

`audio8-codec-bench` can decode captured codes on CPU, Hexagon or OpenCL and
export WAV/raw F32 PCM while measuring the codec separately from generation.
See the [fixed-code decoder benchmark](docs/audio8.md#fixed-code-decoder-benchmark).

`RTF = generation_time / audio_duration`; lower is faster. Dated CI snapshots,
Supertonic 3 and Audio8 campaigns, Apple-silicon measurements, streaming
latency, build pins and reproduction steps live in
[docs/performance.md](docs/performance.md).

CosyVoice3 uses capability-checked flash attention for single-token LM decoding
on CUDA and Metal. Supported CUDA backends reuse decode graphs within masked
128-token cache windows, allowing CUDA graph capture. Unsupported backends or
head dimensions retain their existing attention path.

CosyVoice3 flow uses F16 key/value operands with F32 accumulation for
capability-checked Vulkan flash attention, falling back to F32 operands or
regular attention when required.

CosyVoice3 expands bf16 flow weights to f32 when loading on ARM CPUs. This
avoids scalar bf16 matmuls at twice the resident storage for those weights;
memory preflight includes the expansion. Prefer an f16 flow bundle on ARM.
Use `cosyvoice-bench --llm-gguf`, `--flow-gguf`, and `--hift-gguf` to vary
one stage at a time; see [stage profiling](docs/cosyvoice3.md#stage-profiling).

## Documentation

| Topic | Where |
|---|---|
| Model formats and quantization | [docs/models.md](docs/models.md) |
| APIs, pipelines, per-surface defaults | [docs/api.md](docs/api.md) |
| Voice conditioning (emotion, pace) | [docs/voice-conditioning.md](docs/voice-conditioning.md) |
| Chatterbox | [docs/chatterbox.md](docs/chatterbox.md) |
| Parler-TTS | [docs/parler.md](docs/parler.md) |
| Supertonic | [docs/supertonic.md](docs/supertonic.md) |
| CosyVoice3 | [docs/cosyvoice3.md](docs/cosyvoice3.md) |
| Audio8 | [docs/audio8.md](docs/audio8.md) |
| Pocket TTS | [docs/pocket-tts.md](docs/pocket-tts.md) |
| MOSS Delay, MOSS-SoundEffect, and MOSS-Speech | [docs/moss.md](docs/moss.md) |
| LavaSR enhancement | [docs/lavasr.md](docs/lavasr.md) |
| Build paths and repository layout | [docs/build.md](docs/build.md) |
| CLIs, weight conversion, end-to-end runs | [docs/cli.md](docs/cli.md) |
| Performance deep dive (Apple silicon, CI, streaming) | [docs/performance.md](docs/performance.md) |
| Memory lifecycle (Supertonic and Chatterbox) | [MEMORY.md](MEMORY.md) |
| Backend selection | [docs/backends.md](docs/backends.md) |
| Fixtures and static validation | [docs/testing.md](docs/testing.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Archived port and integration reports | [Chatterbox](docs/history/chatterbox-port.md), [Supertonic](docs/history/supertonic-port.md), [Pocket](docs/history/pocket-integration.md) |
| Voice-clone backward passes | [docs/voiceclone-backward-gap-matrix.md](docs/voiceclone-backward-gap-matrix.md) |

## License

The package code is released under the [MIT License](LICENSE). Models and
conversion-time dependencies retain their own terms: Chatterbox MIT, Pocket
CC-BY-4.0, Supertonic OpenRAIL-M, and Parler/CosyVoice3/Audio8/LavaSR/MOSS
Apache-2.0; see [NOTICE](NOTICE) for
canonical upstream sources and license identities. This in-tree package does
not bundle ggml.
