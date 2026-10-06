"""whisper.linux — Voice typing for Linux desktop."""

# Re-export public API for backward compatibility (tests import from whisper_linux)
import os
import subprocess
import threading
import time

from .config import (
    Config, AppState, CONFIG_DIR, CONFIG_FILE, PID_FILE,
    AVAILABLE_MODELS, DEFAULT_VOICE_COMMANDS, _REPO_ROOT,
    _find_executable, _find_model, _list_models, _list_audio_devices,
    _WHISPER_CLI_NAMES, _WHISPER_SEARCH_DIRS, _MODEL_SEARCH_DIRS,
    LOG_FORMAT, log,
)
from .audio import AudioRecorder, AudioStream, SimpleVAD, _write_wav
from .transcriber import Transcriber, WakeWordDetector, _is_hallucination
from .injector import TextInjector
from .commands import VoiceCommands
from .tray import TrayIcon, _create_icon
from .app import WhisperLinuxApp, send_toggle, main
