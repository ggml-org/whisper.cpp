# Local CLI Setup

All commands below assume you are in the project root (the directory containing `whisper.cpp/`).

## Build whisper.cpp

```bash
cd whisper.cpp
cmake -B build
cmake --build build -j --config Release
```

To enable real-time microphone streaming, install SDL2 first (`brew install sdl2` on macOS, `sudo apt install libsdl2-dev` on Linux) and add `-DWHISPER_SDL2=ON` to the cmake configure step.

## Make whisper-cli available system-wide

```bash
cd whisper.cpp
sudo ln -sf "$(pwd)/build/bin/whisper-cli" /usr/local/bin/whisper-cli
```

After this step, `whisper-cli` is on the PATH. Always invoke it directly as `whisper-cli` — do not reference `build/bin/` paths.

## Download a model

```bash
cd whisper.cpp
sh models/download-ggml-model.sh base.en
```

Available models (English-only, fastest to most accurate):
- `tiny.en` — ~75 MB, fastest, lowest accuracy
- `base.en` — ~142 MB, good default
- `small.en` — ~466 MB, better accuracy
- `medium.en` — ~1.5 GB, high accuracy
- `large-v3` — ~3 GB, multilingual, highest accuracy

## Set environment variable

Add to your shell profile (`~/.zshrc` on macOS, `~/.bashrc` on Linux):

```bash
export WHISPER_CPP_MODEL="/absolute/path/to/whisper.cpp/models/ggml-base.en.bin"
```

Replace `/absolute/path/to` with the actual absolute path to the project root. Then reload your shell.

This is the single source of truth for the model location — always use `$WHISPER_CPP_MODEL`, never hardcode or search for model files.

## Test

Use an absolute path for the audio file:

```bash
whisper-cli -m "$WHISPER_CPP_MODEL" -f "$(pwd)/whisper.cpp/samples/jfk.wav" -np
```

Expected output includes "ask not what your country can do for you".
