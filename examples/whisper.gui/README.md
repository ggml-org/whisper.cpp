# whisper.cpp/examples/whisper.gui

A minimal cross-platform desktop GUI for whisper.cpp, built with
[Dear ImGui](https://github.com/ocornut/imgui) on the SDL2 + OpenGL3 backend.

It is **self-contained / air-gapped friendly**: both Dear ImGui and SDL2 are
vendored under [`deps/`](deps/) and built from source, so a fresh `git clone`
builds with **no network access** and **no SDL2/ImGui system packages**. The
only things the build machine must provide are the OS windowing + OpenGL
*development headers* (X11/Wayland + GL) — these ship with any desktop's base
dev packages and cannot be vendored (OpenGL is your GPU driver).

## Features

- Pick a model and an audio file with a built-in **Browse** dialog (or drag a
  file onto the window — native builds only; see the WSL note below)
- Transcription runs on a **background thread**, so the UI never freezes — with a
  live progress bar and a **Cancel** button
- Timestamped transcript view
- **Speaker diarization** — group segments by voice and label them Speaker 1/2/3 (see below)
- Export the result to **`.txt` / `.srt` / `.json`** (with speaker labels when diarizing)
- Language selection and optional translate-to-English

Supported input formats are the same as the CLI: `wav`, `mp3`, `flac`, `ogg`.
Video files (e.g. `.mp4`) must be converted to audio first — see the
[conversion guide](../cli/README.md#converting-audio--video-to-wav).

## Speaker diarization

Enable **Diarize speakers** to have the transcript split by who is talking. Each
segment is turned into an acoustic fingerprint and the fingerprints are clustered,
so segments spoken by the same voice get the same colored **Speaker N** label
(also written into the exported files).

Set **Speakers** to the known number of people for the best result, or `0` to
auto-detect.

How it works and what to expect — read this before relying on it:

- The fingerprint is built from classic **MFCC features** (no extra model, fully
  offline). This is deliberately **lightweight**, not state-of-the-art.
- It separates **clearly different voices** (e.g. a deep voice vs. a higher one)
  reasonably well, and struggles when voices are **similar**. Auto speaker-count
  is approximate — providing the count helps a lot.
- It is **content-agnostic and pluggable**: the embedding step
  (`diarize::compute_embedding` in [`diarization.cpp`](diarization.cpp)) is the one
  function to replace with a neural speaker-embedding model (e.g. ECAPA-TDNN) for a
  large accuracy jump, without touching the clustering or the UI.

## Building

The GUI is **off by default**. Enable it with `-DWHISPER_GUI=ON`:

```bash
cmake -B build -DWHISPER_GUI=ON
cmake --build build --target whisper-gui -j
./build/bin/whisper-gui
```

This builds the vendored SDL2 statically and the vendored Dear ImGui directly
into the executable. No `FetchContent`, no system SDL2.

> `WHISPER_GUI` is independent of `WHISPER_SDL2` (which links a *system* SDL2 for
> the `stream`/`command`/etc. examples). You do **not** need `WHISPER_SDL2` for
> the GUI.

### Required system development headers

You still need the OS windowing + OpenGL dev headers so SDL2 can compile its
video backend. These are the only things that are not vendored:

```bash
# Fedora / RHEL
sudo dnf install gcc-c++ cmake mesa-libGL-devel libX11-devel libXext-devel \
                 libxkbcommon-devel wayland-devel

# Debian / Ubuntu
sudo apt install g++ cmake libgl1-mesa-dev libx11-dev libxext-dev \
                 libxkbcommon-dev libwayland-dev

# macOS (Xcode command line tools provide OpenGL + Cocoa)
xcode-select --install
```

(The `wayland`/`xkbcommon` packages are optional but recommended so the window
works under Wayland as well as X11.)

## Air-gapped workflow

1. On a **connected** machine, `git clone` this repository (or your fork). The
   clone already contains SDL2 and Dear ImGui under `examples/whisper.gui/deps/`,
   so nothing else needs downloading.
2. Stage the system dev headers above as offline packages (e.g. `dnf download`
   / `apt-get download`) and carry them across with the repo.
3. On the **air-gapped** machine, install those packages, then:
   ```bash
   cmake -B build -DWHISPER_GUI=ON
   cmake --build build --target whisper-gui -j
   ```

The resulting binary links SDL2 statically; at runtime it needs only the OS
OpenGL driver and windowing libraries (`libGL`, `libX11`/Wayland), which are
present on any desktop.

## Running

```bash
./build/bin/whisper-gui
```

Set the model (e.g. `models/ggml-base.en.bin` — fetch one with
`sh ./models/download-ggml-model.sh base.en`), pick an audio file with **Browse**,
and click **Transcribe**. Exported files are written next to the audio file.

> Needs a real display. Over SSH or on WSL, use X/Wayland forwarding or WSLg.

### WSL notes

- **Use the Browse button**, not drag-and-drop: dragging a file from *Windows*
  Explorer into a *Linux* (WSLg) window is not supported by WSLg. The in-app
  Browse dialog has no such limitation.
- **Paths are Linux paths.** A Windows file at `F:\folder\clip.wav` is
  `/mnt/f/folder/clip.wav` here — forward slashes, `/mnt/<drive>/`, and no
  surrounding quotes. (Browse handles this for you, including filenames with
  Unicode characters that are awkward to type.)

## Vendored dependencies

| Path             | Upstream                                            | Version  |
| ---------------- | --------------------------------------------------- | -------- |
| `deps/imgui/`    | [ocornut/imgui](https://github.com/ocornut/imgui)   | v1.91.5  |
| `deps/SDL/`      | [libsdl-org/SDL](https://github.com/libsdl-org/SDL) | 2.30.11  |

The SDL2 tree is trimmed to the library sources (tests, IDE projects and docs
removed) to keep the checkout small.
