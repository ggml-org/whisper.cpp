# Whisper models and formats

Part of the [speech-stack documentation](../README.md).

Whisper and Silero use GGML `.bin` files, not the GGUF format used by the
other engines. Model download, conversion, quantization and Core ML export
follow the [vendored Whisper guide](../third_party/whisper.cpp/README.md).
The supported upstream release is recorded in
[UPSTREAM_PIN](../third_party/whisper.cpp/UPSTREAM_PIN).

## Supported models

The storage tiers below are the documented checkpoint variants. Backend
availability depends on the installed ggml build; Core ML accelerates only
the Whisper encoder. `.en` checkpoints are English-only. Multilingual
checkpoints can translate speech to English; large-v3-turbo is intended for
transcription and is not trained for translation. See the
[upstream language and task guidance](https://github.com/openai/whisper#available-models-and-languages).

| Model | Languages | Parameters | Storage | Backends | Notes |
|---|---|---|---|---|---|
| `whisper-tiny` / `tiny.en` | 99 + translation; `.en`: English only | 39 M | `f16`, `q5_1`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, WebGPU, Core ML |  |
| `whisper-base` / `base.en` | 99 + translation; `.en`: English only | 74 M | `f16`, `q5_1`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, WebGPU, Core ML |  |
| `whisper-small` / `small.en` | 99 + translation; `.en`: English only | 244 M | `f16`, `q5_1`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, WebGPU, Core ML |  |
| `whisper-small.en-tdrz` | English | 244 M | `f16` | CPU, Metal, Vulkan, OpenCL, CUDA | tinydiarize speaker turns |
| `whisper-medium` / `medium.en` | 99 + translation; `.en`: English only | 769 M | `f16`, `q5_0`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML |  |
| `whisper-large-v1` | 99 + translation | 1.55 B | `f16` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML |  |
| `whisper-large-v2` | 99 + translation | 1.55 B | `f16`, `q5_0`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML |  |
| `whisper-large-v3` | 100 + translation | 1.55 B | `f16`, `q5_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML |  |
| `whisper-large-v3-turbo` | 100; transcription | 809 M | `f16`, `q5_0`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | not trained for translation |
| `silero-v5.1.2` | language agnostic | 2 M | `f16` | CPU, Metal, Vulkan, CUDA | voice activity detection; GPU is opt-in via `use_gpu`, default CPU |
| `silero-v6.2.0` | language agnostic | 2 M | `f16` | CPU, Metal, Vulkan, CUDA | voice activity detection; GPU is opt-in via `use_gpu`, default CPU |

## First transcription

From the repository root after the umbrella build:

```sh
bash third_party/whisper.cpp/models/download-ggml-model.sh base.en
./build/bin/whisper-cli -m third_party/whisper.cpp/models/ggml-base.en.bin \
  -f third_party/whisper.cpp/samples/jfk.wav
```

For multi-config generators, use `build/bin/Release/whisper-cli` (with `.exe`
on Windows) after `cmake --build build --config Release`.

## Memory-fit preflight

`whisper-fit-params` (the CLI over `whisper_fit_params` in
[whisper.h](../third_party/whisper.cpp/include/whisper.h)) projects whether a
model, an optional VAD model and a workload fit the free device memory
without reading weight data or allocating anything. `--model` and
`--vad-model` each take the `.bin` or its weightless description: the GGUF
with no data section that the QVAC model registry publishes beside the file,
holding the file's header settings, vocabulary length histogram and tensor
inventory. Both project to the same bytes, so a fit can be answered before
the weights are downloaded. `--verify` measures a real load and needs the
`.bin`.

```sh
./build/bin/whisper-fit-params --model third_party/whisper.cpp/models/ggml-base.en.bin \
  --vad-model third_party/whisper.cpp/models/ggml-silero-v6.2.0.bin --json
./build/bin/whisper-fit-params --model ggml-base.en.fit.gguf --json
```

Run it with `--help` for the workload options.
