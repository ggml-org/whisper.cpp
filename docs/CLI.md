# Command-line tools and examples

Part of the [speech-stack documentation](../README.md).

## Tool selection

The paths below assume a single-config umbrella build from the repository
root. Whisper is under `build/bin/`; the native engines are under
`build/engines/<engine>/`. Multi-config builds add `Release/` or the selected
configuration directory, and Windows executables have `.exe` suffixes.
Standalone output paths are documented in each engine's build guide.

| Binary | Engine | Purpose and full guide |
|---|---|---|
| `whisper-cli` | Whisper | ASR, translation, VAD; [model formats and example](WHISPER.md) |
| `whisper-fit-params` | Whisper | Metadata-only memory projection from a model or its weightless description; [preflight guide](WHISPER.md#memory-fit-preflight) |
| `parakeet` | Parakeet | ASR, diarization, EOU, streaming; [CLI guide](../engines/parakeet/docs/cli.md) |
| `moss-transcribe` | Parakeet | MOSS transcription with speakers, timestamps and hotwords; [MOSS guide](../engines/parakeet/docs/moss-transcribe.md) |
| `parakeet-fit-params` | Parakeet | Metadata-only memory projection; [preflight guide](../engines/parakeet/docs/cli.md#memory-fit-preflight-parakeet-fit-params) |
| `tts-cli` | TTS | Chatterbox batch/streaming; batch Supertonic and limited Parler dispatch; [CLI guide](../engines/tts/docs/cli.md) |
| `supertonic-cli`, `parler-cli`, `cosyvoice-cli`, `audio8-cli` | TTS | Dedicated synthesis options; [API/CLI matrix](../engines/tts/docs/api.md#api-streaming-and-cli-matrix) |
| `pocket-cli` | TTS | Pocket batch WAV and memory preflight; [Pocket guide](../engines/tts/docs/pocket-tts.md) |
| `moss-cli` | TTS | Delay/TTSD synthesis and streaming, `--mode sfx` sound effects, `--mode s2s` spoken replies; [MOSS guide](../engines/tts/docs/moss.md) |
| `chatterbox-fit-params`, `supertonic-fit-params`, `parler-fit-params`, `cosyvoice-fit-params`, `audio8-fit-params` | TTS | Per-model metadata-only projection; [memory-fit APIs](../engines/tts/docs/api.md#memory-fit-preflight) |
| `music-cli` | AudioGen | ACE-Step music generation and editing; [CLI guide](../engines/audiogen/docs/cli.md) |
| `mm3-replay` | AudioGen | MiniMax-Music3 generation/parity; [MiniMax guide](../engines/audiogen/docs/pipeline.md#minimax-music3) |
| `acestep-fit-params` | AudioGen | Metadata-only music memory projection; [preflight guide](../engines/audiogen/docs/cli.md#acestep-fit-params) |
| `acestep-quantize` | AudioGen | ACE-Step/MiniMax stage requantization; [model setup](../engines/audiogen/docs/pipeline.md#model-setup) |
| `acestep-cli`, `lavasr-bench`, `mel2wav` | AudioGen/TTS | Stage validation and enhancement tools; engine guides describe inputs |

Run a tool with `--help` for its complete flag list. An API's incremental
streaming support does not imply its CLI exposes streaming.

## Model setup

Models are acquired separately. Follow the model's guide before running the
examples; placeholder paths below are not provisioned by CMake.

| Engine | Setup |
|---|---|
| Whisper | [GGML model download](WHISPER.md#first-transcription) |
| Parakeet | [Acquisition by family](../engines/parakeet/docs/models.md#acquisition-by-family), including the separate Nemotron and MOSS paths |
| TTS | [Model/storage index](../engines/tts/docs/models.md), linked to each converter and multi-file bundle |
| AudioGen | [ACE-Step model setup](../engines/audiogen/docs/pipeline.md#model-setup), [MiniMax pair](../engines/audiogen/docs/pipeline.md#minimax-music3) |

## Examples

### Whisper

```sh
bash third_party/whisper.cpp/models/download-ggml-model.sh base.en
./build/bin/whisper-cli -m third_party/whisper.cpp/models/ggml-base.en.bin \
  -f third_party/whisper.cpp/samples/jfk.wav
```

Use `whisper-cli --list-devices`, then `--device HTP0` or the exact listed
OpenCL name to select a registered GPU backend. Numeric `--device N` remains
supported; `--no-gpu` excludes GPU backends. `whisper-fit-params` shares the
same selection rules. Without a device flag or `WHISPER_ARG_DEVICE`, both native
tools prefer registered `HTP0`; when it is absent or ambiguous, they retain GPU
index 0. On a Snapdragon
Hexagon build this selects HTP automatically while explicit OpenCL/CPU choices
still override it. See [native Snapdragon setup and validation](WHISPER-HEXAGON.md)
for matching host/DSP builds and execution diagnostics.

### Parakeet

```sh
./build/engines/parakeet/parakeet \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --wav engines/parakeet/test/samples/jfk.wav
```

For attributed transcripts add `--diarization-model` with a compatible
Sortformer or Nemotron 3 Diarization GGUF. Streaming, EOU, language selection
and memory preflight examples live in the [Parakeet CLI guide](../engines/parakeet/docs/cli.md).

### Text-to-speech

```sh
./build/engines/tts/tts-cli \
  --model models/chatterbox-t3-turbo.gguf \
  --s3gen-gguf models/chatterbox-s3gen.gguf \
  --reference-audio me.wav \
  --text "Hello from native C plus plus." --out out.wav
```

Use the dedicated model guides for Supertonic, Parler, CosyVoice, Audio8,
Pocket and MOSS. Supported emotion/pace controls are described in
[voice conditioning](../engines/tts/docs/voice-conditioning.md#voice-conditioning-cross-engine).

### Music generation

```sh
./build/engines/audiogen/music-cli --models models/acestep \
  --caption "Driving synth pop with bright analog leads, 120 bpm" \
  --lyrics "[Instrumental]" --dur 8 --gpu --out song.wav
```

Keep one intended DiT in a scanned model directory or pass `--dit` explicitly.
See the [AudioGen CLI guide](../engines/audiogen/docs/cli.md) for source/reference
audio, editing, backend selection and memory preflight.
