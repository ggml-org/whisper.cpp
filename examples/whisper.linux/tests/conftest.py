"""Pytest fixtures for whisper.linux tests."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path so we can import the 'app' package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Temporary config directory."""
    config_dir = tmp_path / ".config" / "whisper-linux"
    config_dir.mkdir(parents=True)
    return config_dir


@pytest.fixture
def tmp_config_file(tmp_config_dir):
    """Temporary config.ini with valid paths."""
    config_file = tmp_config_dir / "config.ini"
    config_file.write_text(
        "[whisper-linux]\n"
        "whisper_cli = /usr/bin/whisper-cli\n"
        "model = /tmp/ggml-base.bin\n"
        "models_dir = /tmp\n"
        "language = ru\n"
        "threads = 4\n"
        "display_server = x11\n"
        "audio_device = plughw:1,0\n"
        "paste_keys = shift+Insert\n"
        "use_clipboard_fallback = False\n"
        "notification = True\n"
    )
    return config_file


@pytest.fixture
def mock_config(tmp_config_file, monkeypatch):
    """Config that reads from temp config file."""
    import app as wl

    monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_file.parent)
    monkeypatch.setattr(wl.config, "CONFIG_FILE", tmp_config_file)
    return wl.Config()


@pytest.fixture
def mock_config_wayland(mock_config):
    """Config set to Wayland."""
    mock_config.display_server = "wayland"
    return mock_config


@pytest.fixture
def tmp_wav_file():
    """Create a temporary WAV file for testing."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="test-whisper-")
    os.write(fd, b"RIFF" + b"\x00" * 40)  # Minimal WAV header stub
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def tmp_pid_file(tmp_path, monkeypatch):
    """Temporary PID file."""
    import app as wl

    pid_file = tmp_path / "whisper-linux.pid"
    monkeypatch.setattr(wl.config, "PID_FILE", pid_file)
    return pid_file


@pytest.fixture
def mock_config_stream(mock_config):
    """Config set to listen+stream mode (old 'stream' mode)."""
    mock_config.input_mode = "listen"
    mock_config.output_mode = "stream"
    mock_config.wake_word = "дуняша"
    mock_config.silence_timeout = 3.0
    mock_config.vad_threshold = 300
    mock_config.min_speech_ms = 500
    mock_config.max_speech_s = 30.0
    return mock_config


@pytest.fixture
def mock_config_hotkey_stream(mock_config):
    """Config set to hotkey+stream mode."""
    mock_config.input_mode = "hotkey"
    mock_config.output_mode = "stream"
    mock_config.wake_word = "дуняша"
    mock_config.silence_timeout = 3.0
    mock_config.vad_threshold = 300
    mock_config.min_speech_ms = 500
    mock_config.max_speech_s = 30.0
    return mock_config


@pytest.fixture
def mock_config_listen_batch(mock_config):
    """Config set to listen+batch mode."""
    mock_config.input_mode = "listen"
    mock_config.output_mode = "batch"
    mock_config.wake_word = "дуняша"
    mock_config.silence_timeout = 3.0
    mock_config.vad_threshold = 300
    mock_config.min_speech_ms = 500
    mock_config.max_speech_s = 30.0
    return mock_config


@pytest.fixture
def silence_pcm():
    """Return 1 second of silence (s16le mono 16kHz)."""
    return b"\x00\x00" * 16000


@pytest.fixture
def speech_pcm():
    """Return 1 second of loud 'speech' (s16le mono 16kHz, high amplitude sine)."""
    import array, math
    samples = array.array("h")
    for i in range(16000):
        samples.append(int(10000 * math.sin(2 * math.pi * 440 * i / 16000)))
    return samples.tobytes()
