# whisper.linux

Voice typing for Linux desktop using [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

Transcribes speech from microphone and injects text at the cursor position.
Supports X11 and Wayland, hotkey and continuous listening modes, voice commands.

## Requirements

- Python 3.10+
- PyQt5: `pip install PyQt5`
- Built `whisper-cli` from whisper.cpp (see root README)
- A GGML model file (e.g. `ggml-base.bin`)
- `xdotool` (X11/XWayland) or `wtype` (Wayland) for text injection
- `xclip` (X11) or `wl-copy` (Wayland) for clipboard fallback

## Quick Start

```bash
cd examples/whisper.linux

# Start
./whisper-linux

# Start with debug logging
./whisper-linux --debug

# Toggle recording on a running instance
./whisper-linux --toggle

# Stop
pkill -f whisper-linux
```

## CLI Options

```
--toggle           Send toggle signal to running instance
--language LANG    Override language (ru, en, auto)
--model PATH       Override model path
--input-mode MODE  hotkey or listen
--output-mode MODE batch or stream
--stream           Shortcut for --output-mode stream
--wake-word WORD   Override wake word (for listen mode)
--wake-model PATH  Lighter model for wake word detection
--debug            Enable debug logging
```

## Input / Output Modes

Two independent axes control behavior:

| | batch | stream |
|---|---|---|
| **hotkey** | Record all, transcribe, inject at once | Each speech segment transcribed and injected live |
| **listen** | Wake word activates, text accumulated, injected on stop | Wake word activates, each segment injected live |

Default: `hotkey` + `batch` (press hotkey to record, press again to transcribe).

## Voice Commands

When `voice_commands = True` (default), spoken command words trigger key presses
instead of being typed literally. Editable via tray menu: Settings > Edit voice commands.

| Word (EN) | Word (RU) | Action |
|---|---|---|
| enter | энтер, ввод | Press Enter |
| backspace | бэкспейс, назад | Delete previous word |
| tab | таб, табуляция | Press Tab |
| escape, stop | эскейп, стоп | Press Escape |

Commands use fuzzy matching (threshold 0.75), so slight mispronunciations are tolerated.

**Backspace** has special behavior: if there are buffered words not yet injected,
it removes the last word from the buffer. If the buffer is empty, it sends
`Ctrl+BackSpace` to delete the previous word in the editor.

## Keyboard Shortcut

In hotkey mode, you toggle recording via `--toggle`. Set up a global keyboard shortcut
to trigger it from anywhere:

**GNOME** (Settings → Keyboard → Custom Shortcuts → Add):

| Field | Value |
|---|---|
| Name | whisper-linux |
| Command | `/path/to/whisper.linux/whisper-linux --toggle` |
| Shortcut | `Super+V` or any key you prefer |

**Or via command line (GNOME):**

```bash
# Replace /path/to/whisper.linux with the actual path
TOGGLE_CMD="/path/to/whisper.linux/whisper-linux --toggle"

gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings \
  "['/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/whisper-linux/']"

dconf write /org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/whisper-linux/name "'whisper-linux'"
dconf write /org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/whisper-linux/command "'$TOGGLE_CMD'"
dconf write /org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/whisper-linux/binding "'F8'"
```

**KDE:** System Settings → Shortcuts → Custom Shortcuts → Add.

## Configuration

Config file: `~/.config/whisper-linux/config.ini`

All settings are configurable via the system tray menu (right-click the tray icon).

## Autostart

Copy the desktop file to autostart:

```bash
cp whisper-linux.desktop ~/.config/autostart/
```

## Project Structure

```
whisper.linux/
  whisper-linux               # Launcher script (unique process name)
  app/                        # Python package
    __init__.py               # Re-exports public API
    __main__.py               # Entry point
    config.py                 # Config, AppState, constants, helpers
    audio.py                  # AudioRecorder, AudioStream, SimpleVAD
    transcriber.py            # Transcriber, WakeWordDetector
    injector.py               # TextInjector (xdotool/wtype/clipboard)
    commands.py               # VoiceCommands (Enter, Backspace, etc.)
    tray.py                   # TrayIcon, system tray menu, settings
    app.py                    # WhisperLinuxApp, state machine, CLI
  tests/
    conftest.py               # Fixtures
    test_whisper_linux.py     # Tests (all mocked, no hardware needed)
  run_tests.sh                # Run all tests with one command
  whisper-linux.desktop       # Desktop entry for autostart
```

## Running Tests

```bash
# All tests
./run_tests.sh

# With options
./run_tests.sh --debug -x -k "test_toggle"

# Or directly
python3 -m pytest tests/ -v
```
