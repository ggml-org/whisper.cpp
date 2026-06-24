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

The **Diarize** dropdown splits the transcript by who is talking: segments get a
colored **Speaker N** label (also written into the exported files). Set
**Speakers** to the known number of people for the best result, or `0` to
auto-detect. Two modes:

- **Fast (built-in)** — classic **MFCC** clustering, no setup, fully offline and
  dependency-free. Separates clearly different voices but is weak on similar
  voices and noisy/real-world audio. Good for a quick pass.
- **Accurate (sherpa-onnx)** — proper **neural** diarization (a pyannote
  segmentation model + a neural speaker-embedding model). Much better on real
  recordings. The GUI runs the bundled [`diarize.py`](diarize.py) helper for you
  after transcription. One-time setup:

  ```bash
  pip install sherpa-onnx numpy
  ./download-diarization-models.sh        # ~47 MB, into ./models/
  ```

  Launch `whisper-gui` from the repo root (so `examples/whisper.gui/diarize.py`
  and `models/` resolve), choose **Accurate**, set the speaker count, and
  Transcribe. The `diarize.py` field lets you point at the helper if it lives
  elsewhere. If setup is missing, the transcript still appears and the status
  line shows what failed.

The **Fast** embedding step (`diarize::compute_embedding` in
[`diarization.cpp`](diarization.cpp)) is a pluggable seam — but for accuracy,
prefer the **Accurate** mode above.

### Using diarize.py directly (CLI)

You can also run the neural helper outside the GUI on a transcript exported as
JSON (no HuggingFace account needed — models come from GitHub):

```bash
pip install sherpa-onnx numpy
./download-diarization-models.sh        # ~47 MB, into ./models/

# 1) transcribe in the GUI and Export .json   (e.g. audio.json next to audio.wav)
# 2) label it by speaker:
python3 diarize.py audio.wav --json audio.json --speakers 3
#    -> writes audio.spk.txt  (Speaker N: text)
#    -> adds a "speaker" field to audio.json
```

**Set `--speakers N` to the real number of voices** — this is the single most
important knob. Auto-detect (omitting it) tends to over-split into many spurious
speakers. Run `python3 diarize.py -h` for all options. Any WAV is accepted
(it is downmixed and resampled to 16 kHz automatically).

**Hard audio?** For recordings with similar voices or background noise, try a
stronger embedding model. Download one and pass it with `--emb-model`:

```bash
./download-diarization-models.sh models resnet152   # or: titanet-large, campplus, resnet34
python3 diarize.py audio.wav --json audio.json --speakers 4 \
        --emb-model models/wespeaker_en_voxceleb_resnet152_LM.onnx
```

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

### Linux (scripted)

[`whisper-gui-airgap.sh`](whisper-gui-airgap.sh) automates the whole two-stage
flow, including the **neural diarization** dependencies (Python wheels + models),
which the manual steps below do not:

```bash
# 1) on a CONNECTED machine, from the repo root:
examples/whisper.gui/whisper-gui-airgap.sh stage ./whisper-gui-bundle
#    -> bundles the python wheels (sherpa-onnx, numpy), the whisper +
#       diarization models, the OS dev packages (dnf/apt, best-effort), and a
#       copy of the repo.

# 2) copy ./whisper-gui-bundle to the OFFLINE machine, then:
./whisper-gui-bundle/whisper-gui-airgap.sh install ./whisper-gui-bundle
#    -> installs the deps offline, builds whisper-gui, sets up an isolated venv
#       for diarization, and writes run-whisper-gui.sh

# then launch (the launcher wires the venv's python + repo-root cwd):
./whisper-gui-bundle/repo/run-whisper-gui.sh
```

Staging the OS dev packages is the one distro-specific, best-effort part; if it
can't resolve them, install the build deps on the target the normal way (or set
`SKIP_PKGS=1` when staging) — everything else is handled. Wheels are matched to
the staging machine's Python, so stage on a box whose Python minor version
matches the target.

### Linux (manual, build only)

If you only need the GUI (no neural diarization) and will handle deps yourself:

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

### Native Windows, air-gapped (PowerShell)

To build a native Windows `whisper-gui.exe` (no WSL) for an offline target, use
[`whisper-gui-airgap.ps1`](whisper-gui-airgap.ps1). It is a two-stage bundler:

```powershell
# 1) On a CONNECTED Windows machine, from the repo root:
.\examples\whisper.gui\whisper-gui-airgap.ps1 -Mode Stage
#    -> downloads CMake, Python, the wheels (sherpa-onnx, numpy), ffmpeg,
#       the whisper + diarization models, and a copy of the repo into
#       .\whisper-gui-bundle\

# 2) Copy whisper-gui-bundle\ to the OFFLINE machine, then from inside it:
.\whisper-gui-airgap.ps1 -Mode Install
#    -> installs Python + wheels offline, places the models, and builds
#       whisper-gui.exe with MSVC. No network needed.
```

When the build finishes, `-Mode Install` drops a double-clickable
**`WhisperGUI.exe`** at the repo root. It sets the working directory to the repo
root (so the model and `diarize.py` resolve) and starts the GUI — no shell
needed. (Building `whisper-gui` directly also produces it as
`build\bin\Release\WhisperGUI.exe`.)

### Compiler on the target

By default the build uses the target's **Visual Studio 2022 Build Tools**
("Desktop development with C++"). Two ways to avoid needing VS pre-installed:

- **Portable MinGW (recommended, no VS):** stage with `-Toolchain mingw`. This
  bundles a ~150 MB portable MinGW-w64 + Ninja; `-Mode Install` builds with it,
  no Visual Studio required.
  ```powershell
  .\examples\whisper.gui\whisper-gui-airgap.ps1 -Mode Stage -Toolchain mingw
  ```
- **Offline MSVC layout:** stage with `-VsBootstrapper C:\path\to\vs_BuildTools.exe`.
  This runs `--layout` to pre-download the C++ workload (multi-GB) into the
  bundle; `-Mode Install` installs it before building. (The bare
  `vs_BuildTools.exe` is only a downloader — it can't install offline by itself,
  which is why the `--layout` step is required.)

Run `Get-Help .\whisper-gui-airgap.ps1` for all options.

> The native-Windows build path is newer and not yet author-verified on Windows
> — treat the first Stage/Install as a shakedown. On native Windows,
> drag-and-drop **does** work (unlike under WSLg).

## Running

```bash
./build/bin/whisper-gui
```

Set the model (the GUI defaults to `models/ggml-large-v3-turbo.bin` — fetch it
with `sh ./models/download-ggml-model.sh large-v3-turbo`, or grab a smaller one
like `base.en` for low-powered machines), pick an audio file with **Browse**, and
click **Transcribe**. Exported files are written next to the audio file.

> The air-gap bundlers download `large-v3-turbo` by default; pass `-WhisperModel`
> (PowerShell) or `WHISPER_MODEL_NAME=...` (Linux) at the Stage step to bundle a
> different one.

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
