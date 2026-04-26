# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Standard build (Release by default on non-MSVC)
cmake -B build
cmake --build build -j --config Release

# GPU backends
cmake -B build -DGGML_CUDA=1          # NVIDIA CUDA
cmake -B build -DGGML_VULKAN=1        # Vulkan (cross-vendor)
cmake -B build -DGGML_METAL=1         # Apple Metal
cmake -B build -DGGML_BLAS=1          # CPU via OpenBLAS

# Optional features
cmake -B build -DWHISPER_SDL2=ON      # Enable SDL2 for real-time audio (stream example)
cmake -B build -DWHISPER_CURL=ON      # Enable libcurl for model download
cmake -B build -DWHISPER_COREML=ON    # Apple Core ML encoder (Apple Silicon only)

# Sanitizers
cmake -B build -DWHISPER_SANITIZE_ADDRESS=ON
cmake -B build -DWHISPER_SANITIZE_THREAD=ON
cmake -B build -DWHISPER_SANITIZE_UNDEFINED=ON
```

## Running Tests

Tests require model files (downloaded separately) and use CTest:

```bash
# Build with tests enabled (on by default when building standalone)
cmake -B build -DWHISPER_BUILD_TESTS=ON
cmake --build build -j --config Release

# Run all tests
cd build && ctest

# Run a specific test by label
cd build && ctest -L tiny

# Run single integration test manually (requires model at models/for-tests-ggml-tiny.en.bin)
./build/bin/whisper-cli -m models/for-tests-ggml-tiny.en.bin -f samples/jfk.wav

# Run the unit VAD test binary
./build/bin/test-vad
```

## Downloading Models

```bash
# Download a pre-converted ggml model
bash ./models/download-ggml-model.sh base.en   # or: tiny, small, medium, large-v3, etc.

# Download VAD model (for --vad flag)
bash ./models/download-vad-model.sh silero-v6.2.0

# Convenience Makefile targets (downloads model + builds + runs on samples/)
make base.en
make tiny
```

## Transcribing Audio

Audio must be 16-bit WAV at 16 kHz. Convert with ffmpeg:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
./build/bin/whisper-cli -m models/ggml-base.en.bin -f output.wav
```

## Quantization

```bash
./build/bin/quantize models/ggml-base.en.bin models/ggml-base.en-q5_0.bin q5_0
./build/bin/whisper-cli -m models/ggml-base.en-q5_0.bin -f samples/jfk.wav
```

---

## Architecture Overview

### Two-layer design

```
include/whisper.h         ← Public C API (cross-language compatible)
src/whisper.cpp           ← Whisper model: audio preprocessing, encoder, decoder, beam search
src/whisper-arch.h        ← Tensor name map (encoder/decoder/cross-attention weight paths in ggml format)
ggml/                     ← Tensor math library (git subtree from ggml-org/ggml)
```

`whisper_context` holds the loaded model weights (shared, read-only across threads). `whisper_state` holds per-inference mutable state (KV cache, mel buffers). You can create multiple states from one context for parallel inference.

### ggml subdirectory

`ggml/` is a git subtree (synced via `sync` commits). Do not edit it directly unless you are making changes intended to be upstreamed. Hardware backends live in `ggml/src/`:
- `ggml-cpu/` — generic CPU with NEON/AVX/VSX intrinsics
- `ggml-cuda/` — CUDA kernels
- `ggml-metal/` — Metal shaders (Apple)
- `ggml-vulkan/` — Vulkan compute shaders
- `ggml-sycl/` — SYCL (Intel)

### Whisper pipeline (inside `src/whisper.cpp`)

1. **Audio preprocessing** — raw PCM → log-Mel spectrogram (80 mel bins, 30-second chunks at 16 kHz)
2. **Encoder** — convolutional feature extraction + transformer encoder; optional Core ML / OpenVINO offload
3. **Decoder** — autoregressive transformer decoder with optional beam search, temperature fallback, and cross-attention timestamps
4. **VAD** — optional pre-pass using Silero-VAD to skip silence before encoding

### Examples (`examples/`)

Shared utilities used by all examples live at the top level of `examples/`:
- `common.h / common.cpp` — CLI arg parsing, vocab helpers
- `common-whisper.h / common-whisper.cpp` — WAV reading, timestamp formatting
- `common-sdl.h / common-sdl.cpp` — SDL2 audio capture (stream example only)
- `grammar-parser.h / grammar-parser.cpp` — GBNF grammar parsing for constrained decoding

Key example binaries:
| Binary | Source | Purpose |
|--------|--------|---------|
| `whisper-cli` | `examples/cli/` | Primary file transcription tool |
| `whisper-stream` | `examples/stream/` | Real-time mic input (needs SDL2) |
| `whisper-server` | `examples/server/` | HTTP API server |
| `whisper-bench` | `examples/bench/` | Inference benchmarking |
| `quantize` | `examples/quantize/` | Model quantization |
| `vad-speech-segments` | `examples/vad-speech-segments/` | VAD-only segment extraction |

### Bindings (`bindings/`)

Language bindings wrap the C API in `include/whisper.h`:
- `bindings/go/` — Go
- `bindings/java/` — JNI (used by the Android example)
- `bindings/javascript/` — WASM/Node.js (built via Emscripten)
- `bindings/ruby/` — Ruby

### Model format

Models are stored in custom `ggml` binary format (not GGUF). The original OpenAI PyTorch weights are converted with `models/convert-pt-to-ggml.py`. Pre-converted models are available from HuggingFace (`ggerganov/whisper.cpp`). Tensor names follow the pattern defined in `src/whisper-arch.h`.

## Windows-specific notes

The project builds with MSVC. The CMakeLists.txt defines `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` on Windows to work around an MSVC STL issue that causes crashes in the Java bindings. Several MSVC warnings are suppressed project-wide (see the `MSVC_WARNING_FLAGS` block at the bottom of `CMakeLists.txt`).
