# whisper.cpp/examples/whisper.gui

A minimal cross-platform desktop GUI for whisper.cpp, built with
[Dear ImGui](https://github.com/ocornut/imgui) on the SDL2 + OpenGL3 backend.

It stays lightweight: the core inference and audio decoding are reused unchanged,
and Dear ImGui is fetched at configure time (via CMake `FetchContent`) rather than
vendored into the tree.

![overview](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Blank.png)

## Features

- Pick a model and an audio file (or **drag a file onto the window**)
- Transcription runs on a **background thread**, so the UI never freezes — with a
  live progress bar and a **Cancel** button
- Timestamped transcript view
- Export the result to **`.txt` / `.srt` / `.json`**
- Language selection and optional translate-to-English

Supported input formats are the same as the CLI: `wav`, `mp3`, `flac`, `ogg`.
Video files (e.g. `.mp4`) must be converted to audio first — see the
[conversion guide](../cli/README.md#converting-audio--video-to-wav).

## Building

The GUI requires **SDL2** and **OpenGL** development libraries, and network access at
configure time to fetch Dear ImGui.

```bash
# install dependencies (example for Debian/Ubuntu)
sudo apt install libsdl2-dev libgl1-mesa-dev

# configure with SDL2 enabled and build
cmake -B build -DWHISPER_SDL2=ON
cmake --build build --target whisper-gui -j
```

On macOS: `brew install sdl2`. On Windows, provide SDL2 (e.g. via vcpkg) and a
recent MSVC/CMake.

## Running

```bash
./build/bin/whisper-gui
```

Then set the model path (e.g. `models/ggml-base.en.bin`), choose or drag an audio
file, and click **Transcribe**. Exported files are written next to the audio file.

> Offline ImGui fetch: if the build machine has no network, pre-fetch Dear ImGui and
> point CMake at it with `-DFETCHCONTENT_SOURCE_DIR_IMGUI=/path/to/imgui`.
