"""Configuration, constants, and helpers for whisper.linux."""

import configparser
import enum
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("whisper-linux")

# ---------------------------------------------------------------------------
# AppState
# ---------------------------------------------------------------------------

class AppState(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    LISTENING = "listening"      # stream: waiting for wake word
    DICTATING = "dictating"      # stream: actively typing text


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

CONFIG_DIR = Path.home() / ".config" / "whisper-linux"
CONFIG_FILE = CONFIG_DIR / "config.ini"
PID_FILE = Path("/tmp") / "whisper-linux.pid"

_WHISPER_CLI_NAMES = ["whisper-cli", "main"]
_WHISPER_SEARCH_DIRS = [
    _REPO_ROOT / "build" / "bin",
    _REPO_ROOT / "build",
    Path.home() / ".local" / "bin",
    Path("/usr/local/bin"),
    Path("/usr/bin"),
]

_MODEL_SEARCH_DIRS = [
    _REPO_ROOT / "models",
    Path.home() / ".local" / "share" / "whisper-linux" / "models",
]

DEFAULT_VOICE_COMMANDS = {
    "enter": "key:Return",
    "энтер": "key:Return",
    "ввод": "key:Return",
    "backspace": "backspace",
    "бэкспейс": "backspace",
    "бекспейс": "backspace",
    "назад": "backspace",
    "tab": "key:Tab",
    "таб": "key:Tab",
    "табуляция": "key:Tab",
    "escape": "key:Escape",
    "эскейп": "key:Escape",
    "стоп": "key:Escape",
}

AVAILABLE_MODELS = [
    ("tiny",     "~75 MB"),
    ("base",     "~142 MB"),
    ("small",    "~466 MB"),
    ("medium",   "~1.5 GB"),
    ("large-v3", "~3.1 GB"),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _find_executable(names, search_dirs):
    """Find first matching executable in search dirs, then PATH."""
    for d in search_dirs:
        for name in names:
            p = d / name
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
    for name in names:
        for pdir in os.environ.get("PATH", "").split(os.pathsep):
            p = Path(pdir) / name
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
    return None


def _find_model(search_dirs, preferred="ggml-base.bin"):
    """Find first .bin model file."""
    for d in search_dirs:
        p = d / preferred
        if p.is_file():
            return str(p)
    for d in search_dirs:
        if d.is_dir():
            for f in sorted(d.glob("ggml-*.bin")):
                return str(f)
    return None


def _list_models(search_dirs):
    """Return list of (name, path) for all available ggml models, sorted by size."""
    seen = set()
    models = []
    for d in search_dirs:
        if d.is_dir():
            for f in sorted(d.glob("ggml-*.bin")):
                if f.name not in seen:
                    seen.add(f.name)
                    name = f.stem.replace("ggml-", "")
                    models.append((name, str(f)))
    models.sort(key=lambda x: Path(x[1]).stat().st_size)
    return models


def _list_audio_devices():
    """Return list of (label, device_id) for audio input devices."""
    devices = []
    if os.path.isfile("/usr/bin/pw-record"):
        devices.append(("Auto (PipeWire)", "auto"))
    else:
        devices.append(("Auto", "auto"))
    try:
        result = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            m = re.match(r"card\s+(\d+):\s*(\w+).*device\s+(\d+):\s*(.*?)(?:\[|$)", line)
            if m:
                card, card_name, dev, dev_name = m.groups()
                device_id = f"plughw:{card},{dev}"
                label = dev_name.strip() if dev_name.strip() else device_id
                devices.append((label, device_id))
    except Exception:
        pass
    return devices


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    whisper_cli: str = ""
    model: str = ""
    models_dir: str = ""
    language: str = "ru"
    threads: int = 4
    gpu_device: int = -1
    display_server: str = ""
    audio_device: str = ""
    paste_keys: str = "shift+Insert"
    use_clipboard_fallback: bool = False
    notification: bool = True
    input_mode: str = "hotkey"
    output_mode: str = "batch"
    wake_word: str = "марфуша"
    wake_model: str = ""
    silence_timeout: float = 3.0
    vad_threshold: int = 300
    min_speech_ms: int = 500
    max_speech_s: float = 30.0
    end_signal: bool = True
    voice_commands: bool = True

    def __post_init__(self):
        self._gpu_devices = []
        self.voice_commands_map: dict = dict(DEFAULT_VOICE_COMMANDS)
        self._load()

    def _load(self):
        """Load from config.ini, auto-detect missing values."""
        cp = configparser.ConfigParser()
        if CONFIG_FILE.exists():
            cp.read(CONFIG_FILE)

        sec = "whisper-linux"
        if cp.has_section(sec):
            self.whisper_cli = cp.get(sec, "whisper_cli", fallback=self.whisper_cli)
            self.model = cp.get(sec, "model", fallback=self.model)
            self.models_dir = cp.get(sec, "models_dir", fallback=self.models_dir)
            self.language = cp.get(sec, "language", fallback=self.language)
            self.threads = cp.getint(sec, "threads", fallback=self.threads)
            self.gpu_device = cp.getint(sec, "gpu_device", fallback=self.gpu_device)
            self.display_server = cp.get(sec, "display_server", fallback=self.display_server)
            self.audio_device = cp.get(sec, "audio_device", fallback=self.audio_device)
            self.paste_keys = cp.get(sec, "paste_keys", fallback=self.paste_keys)
            self.use_clipboard_fallback = cp.getboolean(
                sec, "use_clipboard_fallback", fallback=self.use_clipboard_fallback
            )
            self.notification = cp.getboolean(sec, "notification", fallback=self.notification)
            old_mode = cp.get(sec, "mode", fallback=None)
            if old_mode and not cp.has_option(sec, "input_mode"):
                if old_mode == "stream":
                    self.input_mode = "listen"
                    self.output_mode = "stream"
                else:
                    self.input_mode = "hotkey"
                    self.output_mode = "batch"
            self.input_mode = cp.get(sec, "input_mode", fallback=self.input_mode)
            self.output_mode = cp.get(sec, "output_mode", fallback=self.output_mode)
            self.wake_word = cp.get(sec, "wake_word", fallback=self.wake_word)
            self.wake_model = cp.get(sec, "wake_model", fallback=self.wake_model)
            self.silence_timeout = cp.getfloat(sec, "silence_timeout", fallback=self.silence_timeout)
            self.vad_threshold = cp.getint(sec, "vad_threshold", fallback=self.vad_threshold)
            self.min_speech_ms = cp.getint(sec, "min_speech_ms", fallback=self.min_speech_ms)
            self.max_speech_s = cp.getfloat(sec, "max_speech_s", fallback=self.max_speech_s)
            self.end_signal = cp.getboolean(sec, "end_signal", fallback=self.end_signal)
            self.voice_commands = cp.getboolean(sec, "voice_commands", fallback=self.voice_commands)

        vc_sec = "voice-commands"
        if cp.has_section(vc_sec):
            saved = dict(cp.items(vc_sec))
            # Merge: keep saved commands, add new defaults that aren't overridden
            merged = dict(DEFAULT_VOICE_COMMANDS)
            merged.update(saved)
            self.voice_commands_map = merged

        if not self.whisper_cli:
            self.whisper_cli = _find_executable(_WHISPER_CLI_NAMES, _WHISPER_SEARCH_DIRS) or "whisper-cli"
        if not self.models_dir:
            self.models_dir = self._detect_models_dir()
        if not self.model:
            self.model = _find_model(self.model_search_dirs) or ""
        if not self.display_server:
            self.display_server = self._detect_display_server()
        if not self.audio_device:
            self.audio_device = self._detect_audio_device()
        detected_gpu = self._detect_gpu_device()
        if self.gpu_device < 0:
            self.gpu_device = detected_gpu

    @staticmethod
    def _detect_display_server():
        xdg = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if "wayland" in xdg:
            return "wayland"
        if "x11" in xdg:
            return "x11"
        if os.environ.get("WAYLAND_DISPLAY"):
            return "wayland"
        if os.environ.get("DISPLAY"):
            return "x11"
        return "x11"

    @staticmethod
    def _detect_audio_device():
        if os.path.isfile("/usr/bin/pw-record"):
            log.info("PipeWire detected (pw-record available), using auto audio device")
            return "auto"
        try:
            result = subprocess.run(
                ["arecord", "-l"], capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                m = re.match(r"card\s+(\d+):.*device\s+(\d+):", line)
                if m:
                    card, dev = m.group(1), m.group(2)
                    device = f"plughw:{card},{dev}"
                    log.info("Auto-detected ALSA audio device: %s (%s)",
                             device, line.strip())
                    return device
        except Exception as e:
            log.warning("Failed to detect audio device: %s", e)
        return "default"

    @staticmethod
    def _detect_models_dir():
        for d in _MODEL_SEARCH_DIRS:
            if d.is_dir() and any(d.glob("ggml-*.bin")):
                return str(d)
        return str(_MODEL_SEARCH_DIRS[0])

    @property
    def model_search_dirs(self):
        dirs = []
        if self.models_dir:
            dirs.append(Path(self.models_dir))
        for d in _MODEL_SEARCH_DIRS:
            if d not in dirs:
                dirs.append(d)
        return dirs

    def _detect_gpu_device(self):
        cli = self.whisper_cli or _find_executable(_WHISPER_CLI_NAMES, _WHISPER_SEARCH_DIRS)
        model = self.model or _find_model(self.model_search_dirs)
        if not cli or not model:
            return 0
        try:
            result = subprocess.run(
                [cli, "-m", model, "-f", "/dev/null"],
                capture_output=True, text=True, timeout=10,
            )
            devices = {}
            for line in result.stderr.splitlines():
                m = re.match(r"ggml_vulkan:\s+(\d+)\s*=\s*(.+?)(?:\s*\|.*)?$", line)
                if m:
                    idx, name = int(m.group(1)), m.group(2).strip()
                    devices[idx] = name
                    log.info("GPU device %d: %s", idx, name)
            self._gpu_devices = [(name, idx) for idx, name in sorted(devices.items())]
            for idx, name in devices.items():
                if "NVIDIA" in name.upper():
                    log.info("Auto-selected NVIDIA GPU device: %d", idx)
                    return idx
            for idx, name in devices.items():
                if "AMD" in name.upper() or "RADEON" in name.upper():
                    log.info("Auto-selected AMD GPU device: %d", idx)
                    return idx
        except Exception as e:
            log.debug("GPU device detection failed: %s", e)
        return 0

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cp = configparser.ConfigParser()
        sec = "whisper-linux"
        cp.add_section(sec)
        cp.set(sec, "whisper_cli", self.whisper_cli)
        cp.set(sec, "model", self.model)
        cp.set(sec, "models_dir", self.models_dir)
        cp.set(sec, "language", self.language)
        cp.set(sec, "threads", str(self.threads))
        cp.set(sec, "gpu_device", str(self.gpu_device))
        cp.set(sec, "display_server", self.display_server)
        cp.set(sec, "audio_device", self.audio_device)
        cp.set(sec, "paste_keys", self.paste_keys)
        cp.set(sec, "use_clipboard_fallback", str(self.use_clipboard_fallback))
        cp.set(sec, "notification", str(self.notification))
        cp.set(sec, "input_mode", self.input_mode)
        cp.set(sec, "output_mode", self.output_mode)
        cp.set(sec, "wake_word", self.wake_word)
        cp.set(sec, "wake_model", self.wake_model)
        cp.set(sec, "silence_timeout", str(self.silence_timeout))
        cp.set(sec, "vad_threshold", str(self.vad_threshold))
        cp.set(sec, "min_speech_ms", str(self.min_speech_ms))
        cp.set(sec, "max_speech_s", str(self.max_speech_s))
        cp.set(sec, "end_signal", str(self.end_signal))
        cp.set(sec, "voice_commands", str(self.voice_commands))
        vc_sec = "voice-commands"
        cp.add_section(vc_sec)
        for word, action in self.voice_commands_map.items():
            cp.set(vc_sec, word, action)
        with open(CONFIG_FILE, "w") as f:
            cp.write(f)
        log.info("Config saved to %s", CONFIG_FILE)
