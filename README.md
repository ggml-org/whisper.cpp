# qvac-ext-lib-whisper.cpp — QVAC speech stack

C++ speech engines on [ggml](https://github.com/tetherto/qvac-ext-ggml): speech-to-text,
speaker diarization, end-of-utterance detection, text-to-speech, and speech
enhancement. Pure C++ inference — no Python or ONNX runtime at runtime — on
CPU and GPU (Metal / Vulkan / OpenCL / CUDA), for desktop (Linux, macOS,
Windows) and mobile (Android, iOS).

## Repo layout

```
CMakeLists.txt              # feature-gated umbrella superbuild (ours)
third_party/whisper.cpp/    # upstream whisper.cpp, vendored as a git subtree
                            #   pinned @ v1.9.1; minimally divergent — every
                            #   QVAC delta is listed in its PATCHES.md
engines/
  parakeet/                 # ASR + diarization + EOU (NVIDIA Parakeet family)
  tts/                      # text-to-speech + speech enhancement
```

## Engines and supported models

| Engine | Task | Models | Docs |
|---|---|---|---|
| [`third_party/whisper.cpp`](third_party/whisper.cpp) | speech-to-text (99 languages), translation | all upstream Whisper GGUFs (`tiny` … `large-v3-turbo`, quantized variants) | [README](third_party/whisper.cpp/README.md) · [PATCHES.md](third_party/whisper.cpp/PATCHES.md) |
| [`engines/parakeet`](engines/parakeet) | speech-to-text (streaming + batch), speaker diarization, end-of-utterance | Parakeet **CTC** 0.6b/1.1b, **TDT** v3/1.1b, **EOU** 120M, **Sortformer** v1 + streaming v2/v2.1 — one `parakeet::Engine` dispatches by GGUF metadata | [README](engines/parakeet/README.md) |
| [`engines/tts`](engines/tts) | text-to-speech (streaming + one-shot), voice cloning, speech enhancement | **Chatterbox** Turbo + Multilingual (23 languages), **Supertonic** v1/v2 (44.1 kHz), **LavaSR** denoiser + Vocos enhancer | [README](engines/tts/README.md) |

All engines share **one ggml**: the `ggml-speech` vcpkg port, built from
[`qvac-ext-ggml`](https://github.com/tetherto/qvac-ext-ggml) (`speech` branch).
The `ggml/` tree vendored inside the whisper subtree is never compiled, and
ggml patches never land in this repo (see the ground rules below).

## Install & build

Prereqs: CMake ≥ 3.20, a C++17 compiler, git. First provision ggml, then the
umbrella:

```sh
# 1) system ggml (same source the ggml-speech vcpkg port ships)
git clone --depth 1 --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src
cmake -S ggml-src -B ggml-src/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_PREFIX=$PWD/ggml-install
cmake --build ggml-src/build -j && cmake --install ggml-src/build

# 2) the speech stack (whisper + parakeet + tts, one shared ggml)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PWD/ggml-install
cmake --build build -j
```

Feature gates: `-DSPEECH_BUILD_{WHISPER,PARAKEET,TTS}=ON/OFF`,
`-DSPEECH_BUILD_TESTS=ON` (then `ctest --test-dir build -LE 'gpu|perf'`),
`-DSPEECH_BUILD_EXECUTABLES=OFF` for library-only builds. GPU backends come
from the ggml build (`-DGGML_VULKAN=ON`, Metal is on by default on Apple).
Each engine also configures standalone (`cmake -S engines/parakeet`, …) —
that's what the per-engine vcpkg ports and CI lanes use.

## Run the models

**Whisper — transcribe a wav:**

```sh
./third_party/whisper.cpp/models/download-ggml-model.sh base.en
./build/bin/whisper-cli -m third_party/whisper.cpp/models/ggml-base.en.bin \
                        -f third_party/whisper.cpp/samples/jfk.wav
```

**Parakeet — transcribe / diarize / detect end-of-utterance** (models are
converted from NeMo checkpoints; see [engines/parakeet/README.md](engines/parakeet/README.md)
for `download-all-models.sh` + `convert-nemo-to-gguf.py`):

```sh
./build/engines/parakeet/parakeet --model models/parakeet-tdt-v3.q8_0.gguf \
                                  --wav engines/parakeet/test/samples/jfk.wav
# the GGUF picks the engine: CTC/TDT/EOU transcription vs Sortformer
# diarization; add --diarization-model <sortformer.gguf> for "who said what"
```

**TTS — text → wav** (GGUF conversion steps in
[engines/tts/README.md](engines/tts/README.md)):

```sh
./build/engines/tts/tts-cli \
    --model      models/chatterbox-t3-turbo.gguf \
    --s3gen-gguf models/chatterbox-s3gen.gguf \
    --text "Hello from native C plus plus." --out /tmp/out.wav
# Multilingual: same flags + --language <code>. A Supertonic GGUF passed as
# --model routes to the 44.1 kHz Supertonic engine automatically.
```

## Performance

Representative end-to-end numbers, taken from the per-engine READMEs (see
those for methodology, hardware, and how to reproduce; `RTF =
inference_time / audio_duration`, lower is better).

**Parakeet** ([full table + methodology](engines/parakeet/README.md#ci-benchmarks-latest-ggml-speech-linux-x86-64),
q8_0, 1 warmup + 5 timed runs, benchmark host `qvac-ubuntu2204-x64-gpu` —
CPU: Intel Core i5-13500, GPU: NVIDIA RTX 4000 SFF Ada, Vulkan):

| Model | CPU RTF (Intel Core i5-13500) | GPU RTF (RTX 4000 SFF Ada) |
|---|--:|--:|
| CTC | 0.078 | 0.0023 |
| TDT | 0.083 | 0.0035 |
| EOU | 0.030 | 0.0052 |
| Sortformer | 0.025 | 0.0020 |

**Chatterbox TTS** ([full table](engines/tts/README.md#performance), Q4_0,
higher `vs real-time` is better):

| Backend / hardware | RTF | vs real-time |
|---|--:|--:|
| Vulkan — NVIDIA RTX 4000 SFF Ada (Linux x86-64) | 0.090 | 11.1× |
| Metal — Mac Studio M3 Ultra | 0.16 | 6.4× |
| CPU — Intel Core i5-13500 (Linux x86-64) | 1.25 | 0.80× |

Supertonic and multilingual Chatterbox numbers (Apple M-series Metal, mobile
GPU paths) live in the [tts performance section](engines/tts/README.md#performance);
whisper benchmarks follow upstream's [`bench`](third_party/whisper.cpp/examples/bench)
methodology. On-device (Android/iOS) performance is tracked by the benchmark
lanes in `tetherto/qvac`.

## Touching `third_party/whisper.cpp`

Don't — unless you're doing an upstream sync (`docs/UPSTREAM-SYNC.md`) or
also updating `third_party/whisper.cpp/PATCHES.md` in the same PR. CI
enforces that the subtree matches the pinned upstream tag outside the
declared patch list. ggml patches never go here; they go to
`qvac-ext-ggml@speech`.

## CI & validation

Per-PR: linux/macos/windows build + non-GPU test lanes per engine, android +
iOS compile smokes, umbrella build with whisper-cli transcription smoke, and
the subtree divergence guard. Engine-affecting changes are additionally
validated downstream in `tetherto/qvac` via an overlay-port draft PR (desktop
integration + Device Farm mobile e2e across all speech addons) before the
registry pin bump.
