# parakeet.cpp

**Parakeet** (NVIDIA FastConformer ASR family, CC-BY-4.0) ported to [`ggml`](https://github.com/ggml-org/ggml). Pure C++ inference on **CPU** and **GPU** (Metal / Vulkan / OpenCL); no Python, PyTorch, or onnxruntime at runtime. One **`parakeet::Engine`** loads **CTC**, **TDT**, **EOU**, or **Sortformer** GGUFs and dispatches by metadata.

## Supported checkpoints

| HF repo | Decoder | Mel | `d_model × n_layers` | Vocab | Params | GGUF size | RTF (Metal) | Languages |
|---|---|---|---|---|---|---|---|---|
| `nvidia/parakeet-ctc-0.6b`    | CTC  | 80  | 1024 × 24 | 1024 | 600 M  | 697 MiB q8_0 / 1.3 GiB f16  | 0.014-0.046 | English only |
| `nvidia/parakeet-ctc-1.1b`    | CTC  | 80  | 1024 × 42 | 1024 | 1.1 B  | 1217 MiB q8_0               | 0.026-0.074 | English only |
| `nvidia/parakeet-tdt-0.6b-v3` | TDT  | 128 | 1024 × 24 | 8192 | 600 M  | 715 MiB q8_0 / 1.34 GiB f16 | 0.006 (q8_0, end-to-end Metal — ~160× realtime, fused LSTM+joint decoder) | ~25 languages + PnC |
| `nvidia/parakeet-tdt-1.1b`    | TDT  | 80  | 1024 × 42 | 1024 | 1.1 B  | 1225 MiB q8_0               | 0.027-0.079 | English only, lowest WER (no PnC) |
| `nvidia/diar_sortformer_4spk-v1` | Sortformer (diarization) | 80 | enc 512 × 18 + tf 192 × 18 | n/a (4 spk) | ~123 M | 263 MiB f16 / 141 MiB q8_0 / 75 MiB q4_0 | 0.017-0.097 | Up to 4 speakers, offline |
| `nvidia/diar_streaming_sortformer_4spk-v2` | Sortformer (diarization) | 128 | enc 512 × 17 + tf 192 × 18 | n/a (4 spk) | ~117 M | 251 MiB f16 / 134 MiB q8_0 / 72 MiB q4_0 | similar to v1 offline | Offline + sliding-history live streaming in-repo; NeMo spkcache-style streaming not implemented |
| `nvidia/parakeet_realtime_eou_120m-v1` | RNN-T + `<EOU>` | 128 | 512 × 17 (chunked-limited att + causal subsampler + LN-in-conv) | 1027 | 120 M | 246 MiB f16 / 132 MiB q8_0 | enc cosine 0.999997 vs NeMo offline; enc on GPU, LSTM decoder CPU-only | English; `<EOU>` turn detection. NVIDIA Open Model License. Offline + Mode 2/3 on fixtures. NeMo `cache_aware_stream_step` path was prototyped and rejected vs offline quality — see `PROGRESS.md`. |

Encoder topology is selected from GGUF metadata (`conv_norm_type`, causal subsampling, chunked-limited attention, etc.), so EOU shares the same C++ graph path as CTC/TDT where weights allow.

## API overview

| Surface | Role |
|---------|------|
| `Engine::transcribe` | One-shot wav → text (CTC / TDT / EOU) or segments (Sortformer) |
| `Engine::transcribe_stream` | Mode 2: full encode once, stream segments |
| `Engine::stream_start` → `StreamSession` | Mode 3: live duplex / cache-aware chunks |
| `Engine::diarize` / `diarize_start` | Sortformer offline / sliding-history live |
| `transcribe_with_speakers` | Sortformer + ASR → attributed transcript |

EOU streaming segments expose `is_eou_boundary`. **`StreamEvent`** (optional callbacks) covers end-of-turn (EOU) and VAD-style signals (Sortformer threshold, optional energy VAD on CTC/TDT). **`Engine::backend_device`** / **`backend_name`** reflect the backend actually used after the load-time cascade.

## Pipeline

```
wav → log-mel → FastConformer encoder → CTC / TDT / EOU / Sortformer decoder
```

Each GGUF bundles weights, mel filterbank, and tokenizer as needed.

## Prerequisites

- C++17, CMake ≥ 3.20  
- Python (torch, `nemo_toolkit[asr]`, `gguf`, numpy, librosa, …) **only** for the scripts under §2 and §4 (`convert-nemo-to-gguf.py`, NeMo reference dumps, and the optional maintainer scripts listed at the end of §4).

## 1. Clone and build

```bash
git clone <this-repo> parakeet.cpp
cd parakeet.cpp
./scripts/setup-ggml.sh

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
```

**GPU backend** — enable exactly **one** at configure time (no runtime switch):

```bash
# Apple Silicon
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON && cmake --build build-metal -j

# Desktop
cmake -S . -B build-vk -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON && cmake --build build-vk -j

# OpenCL (often Adreno; desktop dev may need a vendor/Khronos SDK — see patches/README.md)
cmake -S . -B build-cl -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF   # optional on non-Adreno
cmake --build build-cl -j
```

Run with GPU layers:

```bash
./build/parakeet --n-gpu-layers 1 --model models/parakeet-ctc-0.6b.q8_0.gguf --wav test/samples/jfk.wav
```

`--n-gpu-layers` is a yes/no toggle: any value > 0 offloads the encoder to the compiled GPU backend; on Metal the TDT decoder can run as ggml graphs too. Encoder fits one device; partial-layer offload is not implemented.

**Useful CMake options**

| Flag | Default | Meaning |
|------|---------|---------|
| `PARAKEET_BUILD_LIBRARY` | `ON` | Build the `parakeet` library (linkage follows `BUILD_SHARED_LIBS`; defaults to STATIC when unset) |
| `PARAKEET_BUILD_EXECUTABLES` | `ON` standalone / `OFF` subdir | `parakeet-cli` (binary `parakeet`) |
| `PARAKEET_BUILD_TESTS` | `ON` standalone / `OFF` subdir | `test-*` parity / unit harnesses |
| `PARAKEET_BUILD_EXAMPLES` | `ON` standalone / `OFF` subdir | `live-mic`, `live-mic-attributed` |
| `PARAKEET_INSTALL` | `ON` | Generate `install` rules + the `parakeet-cpp` CMake package config |
| `PARAKEET_USE_SYSTEM_GGML` | `OFF` | Link system ggml instead of `ggml/` submodule |
| `PARAKEET_GGML_LIB_PREFIX` | `ON` | Prefix bundled ggml libs as `speech-ggml-*` (shared with whisper / chatterbox / supertonic so the QVAC speech stack vendors a single ggml file set; no-op when `PARAKEET_USE_SYSTEM_GGML=ON`) |
| `PARAKEET_OPENMP` | `ON` (auto-OFF on Windows non-MinGW) | Try `find_package(OpenMP)` and link the parakeet target against it |
| `PARAKEET_FLASH_ATTN` | `ON` on Metal, `OFF` elsewhere | Fused flash-attn in the encoder MHA (per-backend A/B pending) |
| `PARAKEET_CCACHE` | `ON` | Use ccache as compiler launcher for parakeet targets when found |

With tests enabled, the build emits **`parakeet`** (CLI), **`test-mel`**, **`test-encoder`**, **`test-streaming`**, **`test-vk-vs-cpu`** (if Vulkan), etc. Full list is in CMake / build output.

## 2. Convert weights (`.nemo` → `.gguf`)

```bash
python -m venv venv && . venv/bin/activate
pip install "nemo_toolkit[asr]" gguf numpy soundfile librosa sentencepiece

python scripts/convert-nemo-to-gguf.py \
  --ckpt models/parakeet-ctc-0.6b.nemo \
  --out  models/parakeet-ctc-0.6b.q8_0.gguf
```

**Important:** for non-default checkpoints set **`--hf-repo`** (e.g. `nvidia/parakeet-tdt-0.6b-v3`) — the script otherwise defaults to the CTC repo and may download the wrong weights. Use `scripts/download-all-models.sh` to prefetch `.nemo` files.

Default **`--quant`** is **`q8_0`**. Use **`f16`** for parity-calibrated harnesses (noise from q8 swamps NeMo FP32 references).

### Quantization tiers (CTC 0.6B, M4 Air CPU)

| `--quant` | Size | enc best 20 s | enc best 11 s | Transcript |
|-----------|------|-----------------|---------------|------------|
| `f32` | 2.4 GiB | n/a | n/a | exact |
| `f16` | 1.3 GiB | 1221 ms | ~680 ms | bit-equal |
| `q8_0` | 697 MiB | **839 ms** | **460 ms** | bit-equal |
| `q5_0` | 453 MiB | 1475 ms | ~650 ms | bit-equal |
| `q4_0` | 372 MiB | 1080 ms | 595 ms | bit-equal |

Small tensors and shapes not divisible by 32 may stay f16; see `PROGRESS.md` for quant sweep detail.

### Benchmarks vs onnxruntime (`sample-16k.wav`, 20 s, 5 warmup + 15 runs)

**CPU f16**

```
                   onnxruntime-f16    ggml-cpu-f16
  -----------------------------------------------
  model size           2.3 GiB         1.3 GiB
  load ms              16 736            416      (40x faster)
  inf best ms             948            917      (3 % faster)
  inf median ms         1 007            982      (2 % faster)
  inf stdev ms             52             29      (2x tighter)
  RTF best               0.047          0.046
  RTF median             0.050          0.049
  Transcripts            match          match
```

**CPU int8**

```
                   onnxruntime-int8    ggml-cpu-q8_0
  -------------------------------------------------
  model size          583.9 MiB         697 MiB
  load ms               2 054             359      (5.7x faster)
  inf best ms             677             690      (2 % slower)
  inf median ms           721             715      (1 % faster)
  inf stdev ms             55              16      (3.4x tighter)
  RTF best               0.034           0.040
  RTF median             0.036           0.041
  Transcripts            match           match
```

**Metal vs onnx int8** (same GGUF, `--n-gpu-layers 1`)

```
                   onnxruntime-int8    ggml-metal-q8_0
  ---------------------------------------------------
  model size          583.9 MiB         697 MiB
  load ms               2 295              251      (9.1x faster)
  inf best ms             682              284      (2.4x faster)
  inf median ms           712              286      (2.5x faster)
  inf stdev ms             18             0.55     (33x tighter)
  RTF best               0.034           0.014
  RTF median             0.035           0.014
  Transcripts            match           match
```

On Metal, ggml is ~**2.4–2.5×** faster than onnx int8 with much lower latency variance; encoder runs ~**70×** realtime on the clip. Metal throughput is largely insensitive to quant tier (compute-bound).

## 3. CLI and examples

CMake builds the main binary as target **`parakeet-cli`** with **`OUTPUT_NAME parakeet`** — run **`./build/parakeet`** (path depends on generator). **`parakeet --help`** lists every flag.

### 3.1 `parakeet` (file-based)

**Synopsis:** `parakeet --model <.gguf> (--wav <.wav> | --pcm-in <.raw>) [options]`

The GGUF picks the engine (CTC / TDT / EOU transcription vs Sortformer diarization). Optional **`--diarization-model <sortformer.gguf>`** adds speaker labels when **`--model`** is a CTC/TDT GGUF (“who said what”).

| Topic | Flags |
|------|--------|
| **Input** | **`--model`** (required), **`--wav`** (16 kHz mono), **`--pcm-in`** raw mono PCM, **`--pcm-format`** `s16le` or `f32le`, **`--pcm-rate`** Hz (match model; no resampling) |
| **Compute** | **`--threads N`** (0 = hardware default), **`--n-gpu-layers N`** (>0 = encoder on GPU; yes/no, not partial layers), **`--verbose`** per-stage timings |
| **Streaming** | **`--stream`** → Mode 2 (one full encode, then segments every **`--stream-chunk-ms`**). **`--stream`** + **`--stream-duplex`** → Mode 3 (push chunks; **`--stream-left-context-ms`**, **`--stream-right-lookahead-ms`**, **`--stream-feed-bytes`**). **`--stream-history-ms`** = Sortformer sliding history. **`--emit`** `text` or **`jsonl`** (includes **`is_eou_boundary`** for EOU). |
| **ASR + Sortformer** | **`--diarization-model`**, **`--diarization-min-segment-ms`**, **`--diarization-pad-segment-ms`** |
| **OpenCL** (if compiled in) | **`--opencl-cache-dir`**, **`--opencl-platform`**, **`--opencl-device`**, **`--opencl-disable-fusion`**, **`--opencl-adreno-use-large-buffer`** |
| **Measurements** | **`--bench`** (+ **`--bench-runs`**, **`--bench-warmup`**, **`--bench-json`**), **`--profile`** (+ **`--profile-runs`**, **`--profile-warmup`**), **`--dump-mel PATH`** (raw float32 mel tensor) |
| **Other** | **`--version`**, **`--help`** |

Offline one-shot:

```bash
./build/parakeet --model models/parakeet-ctc-0.6b.q8_0.gguf --wav test/samples/jfk.wav
```

Mode 2 streaming + JSON (EOU shows **`is_eou_boundary`** on the closing chunk when applicable):

```bash
./build/parakeet --model models/parakeet_realtime_eou_120m-v1.q8_0.gguf --wav test/samples/jfk.wav \
  --stream --stream-chunk-ms 1500 --emit jsonl
```

Sortformer sliding-window streaming from file:

```bash
./build/parakeet --model models/diar_sortformer_4spk-v1.f16.gguf \
  --pcm-in speech.raw --pcm-format s16le --pcm-rate 16000 \
  --stream --stream-chunk-ms 2000 --stream-history-ms 30000 --emit text
```

Speaker-attributed transcription (CTC/TDT **`--model`** + Sortformer **`--diarization-model`**):

```bash
./build/parakeet --model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --diarization-model models/diar_sortformer_4spk-v1.f16.gguf \
  --wav test/samples/diarization-sample-16k.wav --emit text
```

Benchmark timing (transcript printed once after stats):

```bash
./build/parakeet --model models/parakeet-ctc-0.6b.q8_0.gguf \
  --wav test/samples/jfk.wav --bench --bench-runs 15 --bench-warmup 5
```

### 3.2 Example programs (microphone)

Enable with **`cmake -DPARAKEET_BUILD_EXAMPLES=ON`**. Produces **`live-mic`** and **`live-mic-attributed`** next to **`parakeet`**. They use **[miniaudio](https://miniaud.io/)** (`examples/miniaudio.h`, capture at **16 kHz mono**). **macOS** prompts for microphone permission on first run; stop with **Ctrl-C** (tail audio is flushed).

| Binary | Purpose |
|--------|---------|
| **`live-mic`** | One GGUF: **CTC/TDT/EOU** → live transcription (**`StreamSession`**); **Sortformer** → live **`[t0-t1] speaker_N`** lines. |
| **`live-mic-attributed`** | Two GGUFs: **`--asr-model`** (CTC/TDT) + **`--diar-model`** (Sortformer) → transcript lines tagged with best-overlap speaker. |

**`live-mic`** (see **`live-mic --help`**):

| Flag | Role |
|------|------|
| **`--model`** | GGUF (required unless **`--list-devices`**) |
| **`--n-gpu-layers`**, **`--threads`** | Same idea as main CLI |
| **`--chunk-ms`** | Transcription segment stride (default **1000**); diarization chunk stride (default **2000**) |
| **`--left-context-ms`**, **`--right-lookahead-ms`** | Transcription Mode 3–style context (defaults **5000** / **1000**) |
| **`--history-ms`** | Diarization sliding history (default **30000**) |
| **`--list-devices`**, **`--device N`** | Capture device selection |
| **`--accumulate`**, **`--silence-flush-ms`** | Transcription: one line until silence or speaker change |
| **`--verbose`** | Forward ggml/backend logs |

```bash
./build/live-mic --list-devices
./build/live-mic --model models/parakeet-ctc-0.6b.q8_0.gguf --n-gpu-layers 1 \
  --chunk-ms 1000 --left-context-ms 5000 --right-lookahead-ms 1000
./build/live-mic --model models/diar_sortformer_4spk-v1.f16.gguf \
  --chunk-ms 2000 --history-ms 30000
```

**`live-mic-attributed`** (see **`live-mic-attributed --help`**):

| Flag | Role |
|------|------|
| **`--asr-model`**, **`--diar-model`** | Required CTC/TDT + Sortformer paths |
| **`--asr-n-gpu-layers`**, **`--diar-n-gpu-layers`** | Independent GPU offload (e.g. ASR on GPU, diar on CPU) |
| **`--asr-chunk-ms`**, **`--asr-left-context-ms`**, **`--asr-right-lookahead-ms`** | Transcription streaming |
| **`--diar-chunk-ms`**, **`--diar-history-ms`** | Diarization streaming |
| **`--speaker-history-ms`** | How much diarization context to keep for attribution (default **60000**) |
| **`--accumulate`**, **`--silence-flush-ms`** | One consolidated line per speaker |

```bash
./build/live-mic-attributed \
  --asr-model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --diar-model models/diar_sortformer_4spk-v1.f16.gguf \
  --asr-chunk-ms 1000 --asr-left-context-ms 5000 --asr-right-lookahead-ms 1000 \
  --diar-chunk-ms 2000 --diar-history-ms 30000
```

## 4. Tests and NeMo parity

```bash
# Parity harnesses need f16 GGUFs + NeMo .npy dumps under artifacts/
python scripts/convert-nemo-to-gguf.py --ckpt models/parakeet-ctc-0.6b.nemo \
  --out models/parakeet-ctc-0.6b.f16.gguf --quant f16
# …same for TDT / Sortformer as needed…

python scripts/dump-ctc-reference.py --wav test/samples/jfk.wav
python scripts/dump-tdt-reference.py --wav test/samples/jfk.wav
python scripts/dump-eou-reference.py --wav test/samples/jfk.wav
python scripts/dump-sortformer-reference.py --wav test/samples/diarization-sample-16k.wav

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Optional maintainer scripts (not required for the workflow above):

| Script | Role |
|--------|------|
| `verify-gguf-roundtrip.py` | Each GGUF tensor vs NeMo `state_dict` after the same layout rules as the converter; catches converter regressions. |
| `ref-encoder-from-gguf.py` | PyTorch encoder from GGUF weights; diff vs `dump-ctc-reference.py` `.npy` outputs to debug layout. |
| `streaming-reference.py` | Chunked CTC with context windows; sanity-check streaming-style output vs offline NeMo. |

Missing fixtures **disable** individual tests (not fail). Labels: `ctest -L unit`, `-L fixture`, `-L perf`, `-L gpu`.

| CMake cache var | Default | Contents |
|-----------------|---------|----------|
| `PARAKEET_TEST_MODEL_DIR` | `models/` | `.gguf` |
| `PARAKEET_TEST_AUDIO_DIR` | `test/samples/` | `.wav` |
| `PARAKEET_TEST_REF_DIR` | `artifacts/` | NeMo `.npy` trees |

**Vulkan:** build with `-DGGML_VULKAN=ON`, run **`test-vk-vs-cpu`** — encoder stages vs CPU, rel tolerances in harness.

Typical f16 stage rel vs NeMo (order of magnitude): mel ~1e-4 inner, blocks ~1e-3, logits ~1e-3, Sortformer probs ~2e-4, EOU encoder cosine ~0.999997. See **`PROGRESS.md`** for quant inflation at q8/q4.

## Current status

- **Shipped:** Offline + Mode 2/3 streaming for CTC/TDT/EOU; Sortformer offline + sliding-history live diarization; optional **`StreamEvent`** callbacks; **`test-vk-vs-cpu`** for Vulkan encoder parity.  
- **Not in-repo:** NeMo-style Sortformer spkcache streaming; KV-cache speedups for Mode 3 (API shape exists).  
- **EOU:** NeMo `cache_aware_stream_step` was evaluated and **rejected** for offline transcript parity — details in **`PROGRESS.md`**.

## Repository layout

| Path | Role |
|------|------|
| `CMakeLists.txt` | Top-level build (library, CLI, tests, examples, install/package config) |
| `cmake/` | Package-config template (`parakeet-cppConfig.cmake.in`) |
| `src/` | Engine, decoders, mel, CLI |
| `include/parakeet/` | Public headers (`parakeet.h`, `engine.h`, `streaming.h`, …) |
| `test/` | `test_*.cpp` CTest sources |
| `examples/` | `live-mic`, `live-mic-attributed`, vendored miniaudio |
| `scripts/` | `setup-ggml.sh`, conversion, NeMo dumps, `download-all-models.sh`; optional tools in §4 |
| `patches/` | ggml patches applied by `setup-ggml.sh` (filename-prefix loader, OpenCL relax, OpenCL kernel-binary cache) |
| `ggml/` | Pinned upstream clone (or `-DPARAKEET_USE_SYSTEM_GGML=ON`) |
| `models/`, `artifacts/`, `test/samples/` | Local fixtures (not tracked) |
| `PROGRESS.md` | Detailed history and parity notes |

## License

Code: [Apache-2.0](LICENSE). Bundled `ggml/`: MIT (`ggml/LICENSE`).

**Weights:** CTC/TDT/Sortformer checkpoints on Hugging Face are **CC-BY-4.0** unless the model card says otherwise; **EOU** (`parakeet_realtime_eou_120m-v1`) uses the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). This repo does not ship weights — download via converter or `download-all-models.sh`.
