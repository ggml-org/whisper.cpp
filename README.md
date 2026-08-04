# qvac-ext-lib-whisper.cpp

On-device speech and audio AI in pure C++ on [ggml](https://github.com/tetherto/qvac-ext-ggml): speech-to-text, speaker diarization, end-of-utterance detection, text-to-speech, voice cloning, speech enhancement, and music generation.

| Property | Value |
|---|---|
| CMake project | `qvac-speech` (feature-gated superbuild over `third_party/` + `engines/`) |
| Runtime dependencies | ggml only. No Python, PyTorch, or ONNX Runtime at inference time |
| Engines | `third_party/whisper.cpp`, `engines/parakeet`, `engines/tts`, `engines/audiogen` |
| Models | every model loads from GGUF (see [Supported models](#supported-models)) |
| Desktop | Linux, macOS, Windows |
| Mobile | Android (arm64-v8a), iOS (arm64) |
| Backends | CPU, Metal, Vulkan, OpenCL (Adreno), CUDA, Apple Core ML (encoder sidecar) |
| Quantization | `f32`, `f16`, `bf16`, `q8_0`, `q6_k`, `q5_0`, `q5_1`, `q4_0` (per model, see tables) |
| Shared ggml | one `ggml-speech` vcpkg port, built from [qvac-ext-ggml@speech](https://github.com/tetherto/qvac-ext-ggml/tree/speech) |
| Language | C++17 |

## Architecture

```
+-----------------------------+  +-----------------------------+
| third_party/whisper.cpp     |  | engines/parakeet            |
| speech-to-text              |  | ASR + diarization + EOU     |
+-----------------------------+  +-----------------------------+
+-----------------------------+  +-----------------------------+
| engines/tts                 |  | engines/audiogen            |
| TTS + cloning + enhancement |  | text-to-music               |
+-----------------------------+  +-----------------------------+
                    |                     :
                    v                     : optional encoder sidecar
   ggml-speech (qvac-ext-ggml@speech)     v
                    |                Apple Core ML
   +--------+-------+-------+---------+
   v        v       v       v         v
  CPU     Metal  Vulkan  OpenCL     CUDA
                        (Adreno)
```

Every component consumes one system ggml, so the whole stack shares a single ggml pin and file set. The `ggml/` tree vendored inside the whisper subtree is never compiled.

## Pipelines

```
whisper   wav  -> log-mel -> encoder -> decoder -> text            (+ Silero VAD, + Core ML encoder)
parakeet  wav  -> log-mel -> FastConformer encoder -> CTC | TDT | EOU | Sortformer
                                                   -> text | speaker segments | turn boundary
tts       text -> LM (T3 / Llama / Qwen2.5) -> acoustic tokens -> CFM or flow -> vocoder -> wav
                                                   (+ LavaSR denoise -> bandwidth extension)
audiogen  caption + lyrics -> text encoder -> ACE-Step LM -> FSQ detokenizer
                           -> DiT flow matching -> Oobleck VAE -> 48 kHz stereo
```

## Repo layout

```
CMakeLists.txt              feature-gated umbrella superbuild
third_party/whisper.cpp/    upstream whisper.cpp, vendored as a git subtree,
                            pinned @ v1.9.1 (f049fff9); every QVAC delta is
                            declared in PATCHES.md and enforced by CI
engines/
  parakeet/                 ASR + diarization + end-of-utterance (NVIDIA Parakeet family)
  tts/                      text-to-speech, voice cloning, speech enhancement
  audiogen/                 music generation (ACE-Step)
docs/UPSTREAM-SYNC.md       how to sync the whisper subtree
```

## Supported models

One row per model. `Backends` lists what the owning engine documents as working; backends not listed are untested for that model, even where ggml would compile them.

### Speech-to-text and translation

| Model | Engine | Languages | Params | Quantization | Backends | Notes |
|---|---|---|---|---|---|---|
| `whisper-tiny` / `tiny.en` | whisper | 99 + translation | 39 M | `f16`, `q5_1`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-base` / `base.en` | whisper | 99 + translation | 74 M | `f16`, `q5_1`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-small` / `small.en` | whisper | 99 + translation | 244 M | `f16`, `q5_1`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-small.en-tdrz` | whisper | English | 244 M | `f16` | CPU, Metal, Vulkan, OpenCL, CUDA | tinydiarize speaker turns |
| `whisper-medium` / `medium.en` | whisper | 99 + translation | 769 M | `f16`, `q5_0`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-large-v1` | whisper | 99 + translation | 1.55 B | `f16` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-large-v2` | whisper | 99 + translation | 1.55 B | `f16`, `q5_0`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-large-v3` | whisper | 99 + translation | 1.55 B | `f16`, `q5_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-large-v3-turbo` | whisper | 99 + translation | 809 M | `f16`, `q5_0`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | fastest large-class decode |
| `silero-v5.1.2` | whisper | language agnostic | 2 M | `f16` | CPU | voice activity detection |
| `silero-v6.2.0` | whisper | language agnostic | 2 M | `f16` | CPU | voice activity detection |
| `nvidia/parakeet-ctc-0.6b` | parakeet | English | 600 M | `f32`, `f16`, `q8_0`, `q5_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, Core ML | offline + streaming + long-form |
| `nvidia/parakeet-ctc-1.1b` | parakeet | English | 1.1 B | `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, Core ML | offline + streaming + long-form |
| `nvidia/parakeet-tdt-0.6b-v3` | parakeet | ~25 + punctuation and capitalization | 600 M | `f32`, `f16`, `q8_0`, `q5_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, Core ML | fused LSTM + joint decoder |
| `nvidia/parakeet-tdt-1.1b` | parakeet | English | 1.1 B | `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, Core ML | lowest WER, no punctuation |

### End-of-utterance and diarization

| Model | Engine | Task | Params | Quantization | Backends | Notes |
|---|---|---|---|---|---|---|
| `nvidia/parakeet_realtime_eou_120m-v1` | parakeet | low-latency ASR + end-of-turn | 120 M | `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL (encoder); CPU (LSTM decoder) | segments expose `is_eou_boundary` |
| `nvidia/diar_sortformer_4spk-v1` | parakeet | diarization, up to 4 speakers | 123 M | `f16`, `q8_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL | offline + sliding-history live |
| `nvidia/diar_streaming_sortformer_4spk-v2` | parakeet | diarization, up to 4 speakers | 117 M | `f16`, `q8_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL | streaming-trained encoder |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | parakeet | diarization, up to 4 speakers | 117 M | `f16`, `q8_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL | Audio-Online Speaker Cache, stable slots across gaps |

Pair any ASR GGUF with a Sortformer GGUF via `--diarization-model` for an attributed "who said what" transcript.

### Text-to-speech and voice cloning

| Model | Engine | Languages | Sample rate | Quantization | Backends | Notes |
|---|---|---|---|---|---|---|
| Chatterbox Turbo | tts | English | 24 kHz | `f16`, `q8_0`, `q5_0`, `q4_0` | CPU, Metal, Vulkan, CUDA | zero-shot voice cloning, 2-step meanflow CFM, streaming |
| Chatterbox Multilingual | tts | 23 | 24 kHz | `f16`, `q8_0`, `q5_0`, `q4_0` | CPU, Metal, Vulkan, CUDA | zero-shot voice cloning, CFG, `--cfm-steps` knob, streaming |
| Supertonic v1 | tts | English | 44.1 kHz | `f32`, `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA | preset voices, streaming |
| Supertonic v2 | tts | 5 (`en`, `ko`, `es`, `pt`, `fr`) | 44.1 kHz | `f32`, `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA | preset voices, streaming |
| Supertonic v3 | tts | 31 + `na` | 44.1 kHz | `f32`, `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA | preset voices, streaming, `na` for unknown source language |
| Parler-TTS mini-v1 | tts | English | 44.1 kHz | `f32`, `f16`, `q8_0`, `q6_k` | CPU, Metal, Vulkan, OpenCL | description-conditioned voice, no cloning |
| Parler-TTS large-v1 | tts | English | 44.1 kHz | `f32`, `f16`, `q8_0`, `q6_k` | CPU, Metal, Vulkan, OpenCL | description-conditioned voice |
| Indic Parler-TTS | tts | 21 Indic | 44.1 kHz | `f32`, `f16`, `q8_0`, `q6_k` | CPU, Metal, Vulkan, OpenCL | Indic prompt BPE tokenizer |
| Fun-CosyVoice3-0.5B | tts | Chinese and English text | 24 kHz | `f32` | CPU | Qwen2.5 LM + DiT flow + CausalHiFT, instruct mode for dialect and emotion |

### Speech enhancement

| Model | Engine | Task | Rate | Quantization | Backends | Notes |
|---|---|---|---|---|---|---|
| LavaSR denoiser (UL-UNAS) | tts | speech denoising | rate preserving, 16 kHz internal STFT | `f32`, `f16` | CPU, OpenCL | applied after synthesis or on captured audio |
| LavaSR enhancer (Vocos BWE) | tts | bandwidth extension | native in, 48 kHz out | `f32`, `f16` | CPU, Metal, Vulkan, OpenCL, CUDA | ConvNeXt + ISTFT head |

### Music generation

| Model | Engine | Task | Rate | Quantization | Backends | Notes |
|---|---|---|---|---|---|---|
| ACE-Step v15 turbo | audiogen | text-to-music | 48 kHz stereo | `f32`, `f16`, `bf16`, `q8_0` | CPU, Vulkan, Metal | 8 diffusion steps by default |
| ACE-Step v15 base / sft | audiogen | text-to-music | 48 kHz stereo | `f32`, `f16`, `bf16`, `q8_0` | CPU, Vulkan, Metal | 50 diffusion steps by default |

## Build

Prerequisites: CMake >= 3.20, a C++17 compiler, git.

```sh
# 1) system ggml (the branch the ggml-speech vcpkg port is cut from; the port
#    pins one commit, so check its portfile REF to match a port build exactly)
git clone --depth 1 --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src
cmake -S ggml-src -B ggml-src/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_PREFIX=$PWD/ggml-install
cmake --build ggml-src/build -j && cmake --install ggml-src/build

# 2) the speech stack (whisper + parakeet + tts + audiogen, one shared ggml)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PWD/ggml-install
cmake --build build -j
```

### CMake options

| Option | Default | Effect |
|---|---|---|
| `SPEECH_BUILD_WHISPER` | `ON` | build `third_party/whisper.cpp` |
| `SPEECH_BUILD_PARAKEET` | `ON` | build `engines/parakeet` |
| `SPEECH_BUILD_TTS` | `ON` | build `engines/tts` |
| `SPEECH_BUILD_AUDIOGEN` | `ON` | build `engines/audiogen` |
| `SPEECH_BUILD_EXECUTABLES` | `ON` | build the CLIs; set `OFF` for library-only builds |
| `SPEECH_BUILD_TESTS` | `OFF` | build the engine test harnesses |
| `SPEECH_BUILD_WHISPER_TESTS` | `OFF` | also build whisper's tests (transcription tests need downloaded models) |

GPU backends come from the ggml build: `-DGGML_VULKAN=ON`, `-DGGML_OPENCL=ON`, `-DGGML_CUDA=ON`; Metal is on by default on Apple. Core ML is gated per engine and defaults to off on both, so add `-DWHISPER_COREML=ON -DPARAKEET_COREML=ON` (Apple only) to the umbrella configure for the Neural Engine encoder paths. For tests, configure with `-DSPEECH_BUILD_TESTS=ON`, then run the non-GPU suite with `ctest --test-dir build -LE 'gpu|perf'`.

Each engine also configures standalone (`cmake -S engines/parakeet`, and so on), which is what the per-engine vcpkg ports and CI lanes use.

### Consumable packages

| vcpkg port | `find_package` | Imported target |
|---|---|---|
| `ggml-speech` | `ggml` | `ggml::ggml` |
| `whisper-cpp` | `whisper` | `whisper::whisper` |
| `parakeet-cpp` | `qvac-parakeet` | `qvac::parakeet` |
| `tts-cpp` | `tts-cpp` | `tts-cpp::tts-cpp` |
| `audiogen-cpp` | `audiogen-cpp` | `audiogen-cpp::audiogen-cpp` |

## Command line tools

| Binary | Engine | Purpose |
|---|---|---|
| `whisper-cli` | whisper | transcribe and translate, with optional Silero VAD |
| `parakeet` | parakeet | transcribe, diarize, detect end-of-utterance, benchmark |
| `tts-cli` | tts | Chatterbox, Supertonic, and Parler synthesis, autodetected from GGUF metadata |
| `parler-cli` | tts | full Parler-TTS flag surface |
| `supertonic-cli` | tts | standalone Supertonic synthesis |
| `cosyvoice-cli` | tts | CosyVoice3 synthesis |
| `music-cli` | audiogen | end-to-end text-to-music |
| `acestep-cli` | audiogen | Oobleck VAE decode and roundtrip harness |
| `lavasr-bench` | tts | denoiser and enhancer benchmark |
| `mel2wav` | tts | HiFT mel to wav |

### Whisper

```sh
./third_party/whisper.cpp/models/download-ggml-model.sh base.en
./build/bin/whisper-cli -m third_party/whisper.cpp/models/ggml-base.en.bin \
                        -f third_party/whisper.cpp/samples/jfk.wav
```

### Parakeet

Models are converted from NeMo checkpoints with `download-all-models.sh` and `convert-nemo-to-gguf.py`; see [engines/parakeet/README.md](engines/parakeet/README.md).

```sh
# transcribe (the GGUF metadata selects CTC / TDT / EOU)
./build/engines/parakeet/parakeet --model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
                                  --wav engines/parakeet/test/samples/jfk.wav

# transcribe with speaker attribution
./build/engines/parakeet/parakeet --model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
                                  --diarization-model models/diar_sortformer_4spk-v1.f16.gguf \
                                  --wav engines/parakeet/test/samples/diarization-sample-16k.wav

# streaming end-of-utterance, JSONL events
./build/engines/parakeet/parakeet --model models/parakeet_realtime_eou_120m-v1.q8_0.gguf \
                                  --wav engines/parakeet/test/samples/jfk.wav \
                                  --stream --stream-chunk-ms 1500 --emit jsonl
```

### Text-to-speech

GGUF conversion steps are in [engines/tts/README.md](engines/tts/README.md).

```sh
# Chatterbox Turbo, with voice cloning from a reference wav
./build/engines/tts/tts-cli --model      models/chatterbox-t3-turbo.gguf \
                            --s3gen-gguf models/chatterbox-s3gen.gguf \
                            --reference-audio me.wav \
                            --text "Hello from native C plus plus." --out out.wav

# Chatterbox Multilingual
./build/engines/tts/tts-cli --model      models/chatterbox-t3-mtl-q4_0.gguf \
                            --s3gen-gguf models/chatterbox-s3gen-mtl-q4_0.gguf \
                            --text "Hola, esto es una demostracion multilingue." \
                            --language es --cfm-steps 7 --out out.wav

# Supertonic, preset voice
./build/engines/tts/tts-cli --model models/supertonic2.gguf --voice M1 --language en \
                            --text "The quick brown fox jumps over the lazy dog." --out out.wav

# Parler-TTS, description-conditioned
./build/engines/tts/parler-cli --model models/parler-mini-v1-q8_0.gguf \
                               --description "A female speaker with a calm, clear voice, close up." \
                               --text "Hey, how are you doing today?" --out out.wav

# CosyVoice3
./build/engines/tts/cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
                                  --text "Hello from a fully on-device pipeline." --out out.wav
```

### Music generation

```sh
./build/engines/audiogen/music-cli --models models/acestep \
                                   --caption "driving synth pop, bright analog leads, 120 bpm" \
                                   --lyrics "[Instrumental]" --dur 8 --gpu --out song.wav
```

## Performance

`RTF = inference_time / audio_duration`, lower is better. The parakeet and tts READMEs carry the full tables, methodology, and reproduction steps; audiogen has no benchmark suite yet and reports per-stage wall clock on stderr.

### ASR, end-of-utterance, diarization

CI numbers, `q8_0` GGUFs, 1 warmup plus 5 timed runs, host `qvac-ubuntu2204-x64-gpu` (CPU: Intel Core i5-13500, GPU: NVIDIA RTX 4000 SFF Ada, Vulkan). Full table: [engines/parakeet/README.md](engines/parakeet/README.md#ci-benchmarks-latest-ggml-speech-linux-x86-64).

| Model | CPU RTF | CPU wall | Vulkan RTF | Vulkan wall |
|---|--:|--:|--:|--:|
| Parakeet CTC | 0.078 | 1572 ms | 0.0023 | 47 ms |
| Parakeet TDT | 0.083 | 1670 ms | 0.0035 | 71 ms |
| Parakeet EOU | 0.030 | 607 ms | 0.0052 | 105 ms |
| Sortformer | 0.025 | 508 ms | 0.0020 | 40 ms |

### Text-to-speech

CI numbers, `q4_0` GGUFs, same host. Full table: [engines/tts/README.md](engines/tts/README.md#performance).

| Model | CPU RTF | Vulkan RTF | Vulkan wall | Vulkan tok/s |
|---|--:|--:|--:|--:|
| Chatterbox Turbo | 1.34 | 0.090 | 368 ms | 186 |
| Chatterbox Multilingual | 4.31 | 0.189 | 1097 ms | 73 |
| Supertonic | 0.079 | n/a | n/a | n/a |

That CI run has no Supertonic GPU lane, so its Vulkan columns are unrecorded rather than unsupported.

### Apple silicon

| Model | Host | Backend | Quantization | RTF | vs real-time |
|---|---|---|---|--:|--:|
| Parakeet TDT 0.6b v3 | Apple silicon, host not recorded | Metal | `q8_0` | 0.006 | 160x |
| Chatterbox Turbo | Mac Studio M3 Ultra | Metal | `q4_0` | 0.16 | 6.4x |
| Chatterbox Turbo | Mac Studio M3 Ultra | CPU (NEON) | `q4_0` | 1.05 | 0.96x |
| Chatterbox Multilingual (`--cfm-steps 7`) | Mac Studio M3 Ultra | Metal | `q4_0` | 0.30 | 3.3x |
| Chatterbox Multilingual | Apple M4 | Metal | `q4_0` | 1.37 | 0.73x |

### Streaming latency

Chatterbox on Apple M4 Metal, 317 speech tokens (12.7 s of audio), `--stream-first-chunk-tokens 10 --stream-chunk-tokens 25 --stream-cfm-steps 1`. Full table: [engines/tts/README.md](engines/tts/README.md#streaming-mode--low-latency-playback).

| Metric | Value |
|---|--:|
| first audio out | 279 ms |
| steady-state chunk RTF | 0.30 to 0.63 |
| overall RTF | 0.90 |

On-device Android and iOS performance is tracked by the benchmark lanes in [QVAC](https://github.com/tetherto/qvac).

## Use in QVAC

These engines ship inside [QVAC](https://github.com/tetherto/qvac) as SDK addons, which consume the vcpkg ports built from this repo. The CLIs here are development and validation entry points: for anything beyond them, such as the JavaScript and TypeScript APIs on the Bare runtime, model download and registry, and desktop plus mobile app integration, see QVAC.

| QVAC addon | Wraps | vcpkg ports consumed |
|---|---|---|
| `@qvac/asr-ggml` | speech-to-text, diarization, end-of-utterance | `whisper-cpp`, `parakeet-cpp` |
| `@qvac/tts-ggml` | text-to-speech, voice cloning, speech enhancement | `tts-cpp` |
| `@qvac/audiogen-ggml` | music generation | `audiogen-cpp` |
| `@qvac/bci-whispercpp` | brain-computer interface transcription | `whisper-cpp` |

## Licenses

| Component | Code license | Model weights |
|---|---|---|
| `third_party/whisper.cpp` | MIT | MIT (OpenAI Whisper), Silero VAD models under their own terms |
| `engines/parakeet` | Apache-2.0 | CC-BY-4.0, except `parakeet_realtime_eou_120m-v1` under the NVIDIA Open Model License |
| `engines/tts` | MIT | Chatterbox MIT, CosyVoice3 Apache-2.0, Supertonic and LavaSR per their model cards |
| `engines/audiogen` | MIT | ACE-Step 1.5 MIT, Qwen3-Embedding Apache-2.0 |

Per-engine `NOTICE` files list every third-party dependency and its license.

## Documentation

| Topic | Where |
|---|---|
| Product using these engines | [QVAC](https://github.com/tetherto/qvac) |
| Speech-to-text engine | [third_party/whisper.cpp/README.md](third_party/whisper.cpp/README.md) |
| Whisper subtree deltas | [third_party/whisper.cpp/PATCHES.md](third_party/whisper.cpp/PATCHES.md) |
| Whisper subtree sync process | [docs/UPSTREAM-SYNC.md](docs/UPSTREAM-SYNC.md) |
| ASR, diarization, end-of-utterance | [engines/parakeet/README.md](engines/parakeet/README.md) |
| Text-to-speech and enhancement | [engines/tts/README.md](engines/tts/README.md) |
| Music generation | [engines/audiogen/README.md](engines/audiogen/README.md) |
| TTS memory behaviour | [engines/tts/MEMORY.md](engines/tts/MEMORY.md) |
| Development journals | [engines/parakeet/PROGRESS.md](engines/parakeet/PROGRESS.md), [engines/tts/PROGRESS.md](engines/tts/PROGRESS.md) |
