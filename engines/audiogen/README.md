# audiogen-cpp

Native [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) and MiniMax-Music3 text-to-music in pure C++ on [ggml](https://github.com/tetherto/qvac-ext-ggml): caption and lyrics in, stereo audio out, no Python or PyTorch at inference time.

| Property | Value |
|---|---|
| CMake project | `audiogen-cpp` v0.1.0 |
| Public API | `tts_cpp::acestep::Engine`, `tts_cpp::minimax::Engine`; memory-fit preflights `tts_cpp::acestep::fit_params`, `tts_cpp::minimax::fit_params` |
| Output | interleaved stereo PCM, model-defined sample rate, `pcm[t * 2 + ch]` |
| Backends | CPU, Vulkan (including Android Mali iGPUs), Metal, OpenCL (validated on Adreno 700+), CUDA, Hexagon NPU on explicit request (`backend = "hexagon"`, every ACE-Step stage including the LM, validated on Snapdragon 8 Elite); WebGPU (Dawn) is selectable with `backend = "webgpu"` but unmeasured, so the LM and detokenizer stay on the CPU there; optional Core ML VAE-decoder sidecar on Apple (`AUDIOGEN_COREML`) |
| ggml | requires the `ggml-speech` port for the custom `ggml_snake` and `ggml_col2im_1d` ops |
| Consumed by | the `@qvac/audiogen-ggml` addon in [QVAC](https://github.com/tetherto/qvac) |

## Supported models

| Model | Task | Output | Quantization | Backends |
|---|---|---|---|---|
| ACE-Step v15 turbo / sft / base | text-to-music, editing, LRC timestamps; base-only lego stems | 48 kHz stereo | `f32`, `f16`, `bf16`, `q8_0`; turbo also `q4_k_m` | CPU, Vulkan, Metal, OpenCL (Adreno 700+), CUDA, Hexagon (turbo `q8_0` DiT); optional Core ML VAE-decoder sidecar |
| MiniMax-Music3 | text-to-music | 44.1 kHz stereo | `f16`, `q8_0`; LM, DiT and depth decoder also `q4_k_m` | desktop CPU + GPU (CUDA, Vulkan, Metal); optional Core ML conditioner projection on Apple |

Apple builds can optionally use a Core ML sidecar for the ACE-Step VAE decoder.
Export, placement and fallback contracts are in
[docs/backends.md](docs/backends.md#core-ml-vae-decoder-sidecar).
MiniMax-Music3 can explicitly use a [Core ML conditioner projection](docs/backends.md#minimax-core-ml-conditioner)
with its remaining stages on the configured native backend. MiniMax is unavailable on mobile.

## Performance

Dated ACE-Step and MiniMax campaigns, including CUDA/Vulkan/Metal and
Snapdragon measurements, live in [docs/performance.md](docs/performance.md).
For reproducible comparisons with upstream `acestep.cpp`, use the
[comparison harness](benchmarks/comparison/README.md).

The [music alignment guide](../../scripts/benchmarks/music-alignment.md)
documents opt-in CLAP diagnostics, artifacts, benchmark timing boundaries,
and MiniMax model preparation and transfer. Diagnostics do not gate quality.

## Documentation

| Topic | Where |
|---|---|
| Pipeline, model stages, model setup | [docs/pipeline.md](docs/pipeline.md) |
| Backends, environment overrides, Core ML VAE sidecar | [docs/backends.md](docs/backends.md) |
| Build | [docs/build.md](docs/build.md) |
| Audio editing (repaint, FlowEdit) | [docs/editing.md](docs/editing.md) |
| Command-line tools | [docs/cli.md](docs/cli.md) |
| Memory-fit preflight | [docs/memory-fit.md](docs/memory-fit.md) |
| Tests and parity | [docs/testing.md](docs/testing.md) |
| Performance campaigns and timing methods | [docs/performance.md](docs/performance.md) |
| Engine comparison harness | [benchmarks/comparison/README.md](benchmarks/comparison/README.md) |

## License

`audiogen-cpp` is MIT licensed; see [LICENSE](LICENSE). Model weights and upstream model code carry their own terms, listed in [NOTICE](NOTICE).
