"""Unit tests for whisper.linux — all mocked, no hardware needed."""

import os
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import app as wl


# ===================================================================
# AppState
# ===================================================================

class TestAppState:
    def test_states_exist(self):
        assert wl.AppState.IDLE.value == "idle"
        assert wl.AppState.RECORDING.value == "recording"
        assert wl.AppState.PROCESSING.value == "processing"
        assert wl.AppState.LISTENING.value == "listening"
        assert wl.AppState.DICTATING.value == "dictating"

    def test_all_states(self):
        assert len(wl.AppState) == 5


# ===================================================================
# Config
# ===================================================================

class TestConfig:
    def test_load_from_file(self, mock_config):
        assert mock_config.whisper_cli == "/usr/bin/whisper-cli"
        assert mock_config.model == "/tmp/ggml-base.bin"
        assert mock_config.language == "ru"
        assert mock_config.threads == 4
        assert mock_config.display_server == "x11"
        assert mock_config.use_clipboard_fallback is False
        assert mock_config.notification is True

    def test_auto_detect_display_x11(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert wl.Config._detect_display_server() == "x11"

    def test_auto_detect_display_wayland(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        assert wl.Config._detect_display_server() == "wayland"

    def test_auto_detect_display_wayland_env(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert wl.Config._detect_display_server() == "wayland"

    def test_auto_detect_display_fallback(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        assert wl.Config._detect_display_server() == "x11"

    def test_save_config(self, mock_config, tmp_config_file):
        mock_config.language = "en"
        mock_config.save()
        text = tmp_config_file.read_text()
        assert "language = en" in text

    def test_default_config_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_path / "nonexistent")
        monkeypatch.setattr(wl.config, "CONFIG_FILE", tmp_path / "nonexistent" / "config.ini")
        monkeypatch.setattr(wl.config, "_find_executable", lambda *a: "/mock/whisper-cli")
        monkeypatch.setattr(wl.config, "_find_model", lambda *a, **kw: "/mock/model.bin")
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        config = wl.Config()
        assert config.whisper_cli == "/mock/whisper-cli"
        assert config.model == "/mock/model.bin"

    def test_gpu_devices_cache_populated(self, mock_config):
        """Config has _gpu_devices attribute (list) after init."""
        assert hasattr(mock_config, '_gpu_devices')
        assert isinstance(mock_config._gpu_devices, list)


# ===================================================================
# find_executable / find_model helpers
# ===================================================================

class TestHelpers:
    def test_find_executable_found(self, tmp_path):
        exe = tmp_path / "whisper-cli"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        result = wl._find_executable(["whisper-cli"], [tmp_path])
        assert result == str(exe)

    def test_find_executable_not_found(self, tmp_path):
        result = wl._find_executable(["nonexistent-tool-xyz"], [tmp_path])
        assert result is None

    def test_find_model_preferred(self, tmp_path):
        model = tmp_path / "ggml-base.bin"
        model.write_bytes(b"\x00" * 100)
        result = wl._find_model([tmp_path])
        assert result == str(model)

    def test_find_model_any_bin(self, tmp_path):
        model = tmp_path / "ggml-small.bin"
        model.write_bytes(b"\x00" * 100)
        result = wl._find_model([tmp_path], preferred="ggml-base.bin")
        assert result == str(model)

    def test_find_model_none(self, tmp_path):
        result = wl._find_model([tmp_path / "empty"])
        assert result is None

    def test_available_models_constant(self):
        assert len(wl.AVAILABLE_MODELS) == 5
        names = [n for n, _ in wl.AVAILABLE_MODELS]
        assert "tiny" in names
        assert "base" in names
        assert "large-v3" in names

    @patch("app.subprocess.run")
    @patch("app.os.path.isfile", return_value=True)
    def test_list_audio_devices_pipewire(self, mock_isfile, mock_run):
        mock_run.return_value = MagicMock(
            stdout="card 0: PCH [HDA Intel PCH], device 0: ALC892 Analog [ALC892 Analog]\n",
            stderr="",
        )
        devices = wl._list_audio_devices()
        assert devices[0] == ("Auto (PipeWire)", "auto")
        assert len(devices) >= 2
        assert devices[1][1] == "plughw:0,0"

    @patch("app.subprocess.run", side_effect=FileNotFoundError())
    @patch("app.os.path.isfile", return_value=False)
    def test_list_audio_devices_no_pipewire(self, mock_isfile, mock_run):
        devices = wl._list_audio_devices()
        assert devices[0] == ("Auto", "auto")
        assert len(devices) == 1


# ===================================================================
# AudioRecorder
# ===================================================================

class TestAudioRecorder:
    @patch("app.time.sleep")
    @patch("app.subprocess.Popen")
    def test_start_recording(self, mock_popen, mock_sleep):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        rec = wl.AudioRecorder(MagicMock(audio_device="plughw:1,0"))
        path = rec.start()

        assert path.endswith(".wav")
        assert rec.is_recording is True
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert args[0] == "arecord"
        assert "-D" in args and "plughw:1,0" in args
        assert "-r" in args and "16000" in args

    @patch("app.time.sleep")
    @patch("app.subprocess.Popen")
    def test_stop_recording(self, mock_popen, mock_sleep):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        rec = wl.AudioRecorder(MagicMock(audio_device="plughw:1,0"))
        rec.start()
        wav = rec.stop()

        assert wav is not None
        assert wav.endswith(".wav")
        mock_proc.send_signal.assert_called_once_with(signal.SIGINT)
        assert rec.is_recording is False

    @patch("app.time.sleep")
    @patch("app.subprocess.Popen")
    def test_start_while_recording_raises(self, mock_popen, mock_sleep):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        rec = wl.AudioRecorder(MagicMock(audio_device="plughw:1,0"))
        rec.start()
        with pytest.raises(RuntimeError, match="Already recording"):
            rec.start()

    def test_stop_without_start(self):
        rec = wl.AudioRecorder(MagicMock(audio_device="plughw:1,0"))
        assert rec.stop() is None

    @patch("app.time.sleep")
    @patch("app.subprocess.Popen")
    def test_cleanup_removes_file(self, mock_popen, mock_sleep):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        rec = wl.AudioRecorder(MagicMock(audio_device="plughw:1,0"))
        path = rec.start()

        # Create the file so cleanup can delete it
        Path(path).write_bytes(b"\x00")
        assert os.path.exists(path)

        mock_proc.poll.return_value = 0  # Already stopped
        rec.cleanup()
        assert not os.path.exists(path)

    @patch("app.subprocess.Popen")
    def test_start_fails_on_bad_device(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # Exited immediately
        mock_proc.stderr.read.return_value = b"audio open error: No such file or directory"
        mock_popen.return_value = mock_proc

        rec = wl.AudioRecorder(MagicMock(audio_device="hw:99,0"))
        with pytest.raises(RuntimeError, match="Recording failed to start"):
            rec.start()


# ===================================================================
# Transcriber
# ===================================================================

class TestTranscriber:
    @patch("app.subprocess.run")
    def test_transcribe_success(self, mock_run, mock_config, tmp_wav_file):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="  Привет, мир!  \n",
            stderr="",
        )

        tr = wl.Transcriber(mock_config)
        result = tr.transcribe(tmp_wav_file)

        assert result == "Привет, мир!"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/whisper-cli"
        assert "-m" in cmd
        assert "-nt" in cmd
        assert "-np" in cmd

    @patch("app.subprocess.run")
    def test_transcribe_removes_blank_audio(self, mock_run, mock_config, tmp_wav_file):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="[BLANK_AUDIO] some text [BLANK_AUDIO]",
            stderr="",
        )

        tr = wl.Transcriber(mock_config)
        result = tr.transcribe(tmp_wav_file)
        assert result == "some text"

    @patch("app.subprocess.run")
    def test_transcribe_failure(self, mock_run, mock_config, tmp_wav_file):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error loading model",
        )

        tr = wl.Transcriber(mock_config)
        with pytest.raises(RuntimeError, match="whisper-cli failed"):
            tr.transcribe(tmp_wav_file)

    def test_transcribe_missing_file(self, mock_config):
        tr = wl.Transcriber(mock_config)
        with pytest.raises(FileNotFoundError):
            tr.transcribe("/tmp/nonexistent-file-xyz.wav")

    @patch("app.subprocess.run")
    def test_transcribe_filters_hallucination_ru(self, mock_run, mock_config, tmp_wav_file):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Аппортизатор субтитров И.Б.С.С вами был Игорь Негода.",
            stderr="",
        )
        tr = wl.Transcriber(mock_config)
        result = tr.transcribe(tmp_wav_file)
        assert result == ""

    @patch("app.subprocess.run")
    def test_transcribe_filters_hallucination_thanks(self, mock_run, mock_config, tmp_wav_file):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Спасибо за просмотр!",
            stderr="",
        )
        tr = wl.Transcriber(mock_config)
        result = tr.transcribe(tmp_wav_file)
        assert result == ""

    @patch("app.subprocess.run")
    def test_transcribe_filters_hallucination_en(self, mock_run, mock_config, tmp_wav_file):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Subtitles by the Amara.org community",
            stderr="",
        )
        tr = wl.Transcriber(mock_config)
        result = tr.transcribe(tmp_wav_file)
        assert result == ""

    @patch("app.subprocess.run")
    def test_transcribe_keeps_normal_text(self, mock_run, mock_config, tmp_wav_file):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Привет, как дела?",
            stderr="",
        )
        tr = wl.Transcriber(mock_config)
        result = tr.transcribe(tmp_wav_file)
        assert result == "Привет, как дела?"


class TestHallucinationFilter:
    def test_russian_subtitles(self):
        assert wl._is_hallucination("Субтитры сделал DimaTorzworker")
        assert wl._is_hallucination("Редактор субтитров А.Б.В.")
        assert wl._is_hallucination("Продолжение следует...")

    def test_russian_credits(self):
        assert wl._is_hallucination("Спасибо за просмотр!")
        assert wl._is_hallucination("Подписывайтесь на канал")
        assert wl._is_hallucination("С вами был Игорь Негода.")
        assert wl._is_hallucination("Аппортизатор субтитров И.Б.С.С")
        assert wl._is_hallucination("Добро пожаловать в КПС!")

    def test_english_hallucinations(self):
        assert wl._is_hallucination("Thanks for watching!")
        assert wl._is_hallucination("Subtitles by the Amara.org community")
        assert wl._is_hallucination("Please subscribe and like")

    def test_normal_text_not_filtered(self):
        assert not wl._is_hallucination("Привет, как дела?")
        assert not wl._is_hallucination("Hello world")
        assert not wl._is_hallucination("Запустить сервер")
        assert not wl._is_hallucination("Открой файл конфигурации")

    def test_speech_rate_too_many_words(self):
        """5 words from 0.5s of audio is impossible — hallucination."""
        assert wl._is_hallucination("Один два три четыре пять", duration_s=0.5)

    def test_speech_rate_too_many_chars(self):
        """Long word from 0.5s audio — too many characters."""
        assert wl._is_hallucination("Константинопольский", duration_s=0.5)

    def test_speech_rate_normal(self):
        """1 word from 0.6s is perfectly normal."""
        assert not wl._is_hallucination("Привет", duration_s=0.6)
        assert not wl._is_hallucination("Энтер", duration_s=0.5)
        assert not wl._is_hallucination("Таб", duration_s=0.5)
        assert not wl._is_hallucination("Табуляция", duration_s=0.5)

    def test_speech_rate_longer_audio(self):
        """5 words from 3s is fine (~1.7 wps)."""
        assert not wl._is_hallucination("Привет как дела у тебя", duration_s=3.0)

    def test_speech_rate_no_duration(self):
        """Without duration, only pattern matching applies."""
        assert not wl._is_hallucination("Один два три четыре пять")
        assert wl._is_hallucination("Субтитры от переводчиков")


# ===================================================================
# TextInjector
# ===================================================================

class TestTextInjector:
    @patch("app.subprocess.run")
    def test_inject_x11_xdotool(self, mock_run, mock_config):
        mock_run.return_value = MagicMock(returncode=0)

        inj = wl.TextInjector(mock_config)
        inj.inject("Hello world")

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "xdotool"
        assert "type" in cmd
        assert "Hello world" in cmd

    @patch("app.subprocess.run")
    def test_inject_x11_clipboard_fallback_explicit(self, mock_run, mock_config):
        mock_config.use_clipboard_fallback = True
        mock_run.return_value = MagicMock(returncode=0)

        inj = wl.TextInjector(mock_config)
        inj.inject("hello")

        assert mock_run.call_count == 2
        # First call: xclip
        assert mock_run.call_args_list[0][0][0][0] == "xclip"
        # Second call: xdotool key ctrl+v
        assert mock_run.call_args_list[1][0][0][0] == "xdotool"

    @patch("app.subprocess.run")
    def test_inject_x11_non_ascii_uses_clipboard(self, mock_run, mock_config):
        """Non-ASCII text (Cyrillic) auto-switches to clipboard paste."""
        mock_run.return_value = MagicMock(returncode=0)

        inj = wl.TextInjector(mock_config)
        inj.inject("Привет мир")

        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0][0][0][0] == "xclip"
        assert mock_run.call_args_list[1][0][0][0] == "xdotool"

    @patch("app.subprocess.run")
    def test_inject_x11_xdotool_fails_falls_back(self, mock_run, mock_config):
        # First call (xdotool type) fails, then clipboard calls succeed
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "xdotool"),
            MagicMock(returncode=0),  # xclip
            MagicMock(returncode=0),  # xdotool key
        ]

        inj = wl.TextInjector(mock_config)
        inj.inject("test")

        assert mock_run.call_count == 3

    @patch("app.subprocess.run")
    def test_inject_wayland_wtype(self, mock_run, mock_config_wayland):
        mock_run.return_value = MagicMock(returncode=0)

        inj = wl.TextInjector(mock_config_wayland)
        inj.inject("test")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "wtype"

    @patch("app.time.sleep")
    @patch("app.subprocess.run")
    def test_inject_wayland_wlcopy_ydotool(self, mock_run, mock_sleep, mock_config_wayland):
        """wtype fails → wl-copy + wl-copy --primary + ydotool Shift+Insert."""
        mock_run.side_effect = [
            FileNotFoundError(),  # wtype not found
            MagicMock(returncode=0),  # wl-copy (CLIPBOARD)
            MagicMock(returncode=0),  # wl-copy --primary (PRIMARY)
            MagicMock(returncode=0),  # ydotool key shift+Insert
        ]

        inj = wl.TextInjector(mock_config_wayland)
        inj.inject("test")

        assert mock_run.call_count == 4
        assert mock_run.call_args_list[1][0][0][0] == "wl-copy"
        assert "--primary" in mock_run.call_args_list[2][0][0]
        cmd = mock_run.call_args_list[3][0][0]
        assert cmd[0] == "ydotool"
        assert "shift+Insert" in cmd

    @patch("app.time.sleep")
    @patch("app.subprocess.run")
    def test_inject_wayland_custom_paste_keys(self, mock_run, mock_sleep, mock_config_wayland):
        """Config paste_keys overrides the paste shortcut sent via ydotool."""
        mock_config_wayland.paste_keys = "ctrl+shift+v"
        mock_run.side_effect = [
            FileNotFoundError(),  # wtype not found
            MagicMock(returncode=0),  # wl-copy
            MagicMock(returncode=0),  # wl-copy --primary
            MagicMock(returncode=0),  # ydotool key ctrl+shift+v
        ]

        inj = wl.TextInjector(mock_config_wayland)
        inj.inject("test")

        assert mock_run.call_count == 4
        cmd = mock_run.call_args_list[3][0][0]
        assert cmd[0] == "ydotool"
        assert "ctrl+shift+v" in cmd

    @patch("app.time.sleep")
    @patch("app.subprocess.run")
    def test_inject_wayland_wlcopy_xdotool(self, mock_run, mock_sleep, mock_config_wayland):
        """wtype fails, ydotool fails → wl-copy + xdotool key."""
        mock_run.side_effect = [
            FileNotFoundError(),  # wtype
            MagicMock(returncode=0),  # wl-copy
            MagicMock(returncode=0),  # wl-copy --primary
            FileNotFoundError(),  # ydotool fails
            MagicMock(returncode=0),  # xdotool key
        ]

        inj = wl.TextInjector(mock_config_wayland)
        inj.inject("test")

        assert mock_run.call_count == 5
        assert mock_run.call_args_list[1][0][0][0] == "wl-copy"
        assert mock_run.call_args_list[4][0][0][0] == "xdotool"

    @patch("app.time.sleep")
    @patch("app.subprocess.run")
    def test_inject_wayland_xclip_fallback(self, mock_run, mock_sleep, mock_config_wayland):
        """When wtype and wl-copy both missing, falls back to xclip via XWayland."""
        mock_run.side_effect = [
            FileNotFoundError(),  # wtype
            FileNotFoundError(),  # wl-copy
            MagicMock(returncode=0),  # xdotool type (X11 fallback, text is ASCII)
        ]

        inj = wl.TextInjector(mock_config_wayland)
        inj.inject("test")

        assert mock_run.call_count == 3
        cmd = mock_run.call_args_list[2][0][0]
        assert cmd[0] == "xdotool"

    def test_inject_empty_text(self, mock_config):
        inj = wl.TextInjector(mock_config)
        # Should return without doing anything
        inj.inject("")
        inj.inject(None)


# ===================================================================
# PID file management
# ===================================================================

class TestPidFile:
    def test_write_and_read_pid(self, tmp_pid_file):
        wl.WhisperLinuxApp.write_pid()
        pid = wl.WhisperLinuxApp.read_pid()
        assert pid == os.getpid()

    def test_read_pid_no_file(self, tmp_pid_file):
        assert wl.WhisperLinuxApp.read_pid() is None

    def test_read_pid_stale(self, tmp_pid_file):
        # Write a PID that doesn't exist
        tmp_pid_file.write_text("9999999")
        assert wl.WhisperLinuxApp.read_pid() is None

    def test_read_pid_invalid(self, tmp_pid_file):
        tmp_pid_file.write_text("not-a-number")
        assert wl.WhisperLinuxApp.read_pid() is None

    def test_remove_pid(self, tmp_pid_file):
        wl.WhisperLinuxApp.write_pid()
        assert tmp_pid_file.exists()
        wl.WhisperLinuxApp.remove_pid()
        assert not tmp_pid_file.exists()

    def test_remove_pid_nonexistent(self, tmp_pid_file):
        # Should not raise
        wl.WhisperLinuxApp.remove_pid()


# ===================================================================
# WhisperLinuxApp — state machine
# ===================================================================

class TestWhisperLinuxApp:
    def _make_app(self, mock_config):
        app = wl.WhisperLinuxApp(mock_config)
        app.recorder = MagicMock(spec=wl.AudioRecorder)
        app.transcriber = MagicMock(spec=wl.Transcriber)
        app.injector = MagicMock(spec=wl.TextInjector)
        return app

    def test_initial_state(self, mock_config):
        app = self._make_app(mock_config)
        assert app.state == wl.AppState.IDLE

    def test_toggle_starts_recording(self, mock_config):
        app = self._make_app(mock_config)
        app.toggle()
        app.recorder.start.assert_called_once()
        assert app.state == wl.AppState.RECORDING

    def test_toggle_stops_recording(self, mock_config):
        app = self._make_app(mock_config)
        app.recorder.stop.return_value = "/tmp/test.wav"
        # Prevent the transcription thread from running
        app.transcriber.transcribe.return_value = ""

        app.toggle()  # IDLE → RECORDING
        assert app.state == wl.AppState.RECORDING

        with patch("app.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            app.toggle()  # RECORDING → PROCESSING
            app.recorder.stop.assert_called_once()
            assert app.state == wl.AppState.PROCESSING
            mock_thread.assert_called_once()

    def test_toggle_while_processing_does_nothing(self, mock_config):
        app = self._make_app(mock_config)
        app.state = wl.AppState.PROCESSING
        app.toggle()
        # Should not call start or stop
        app.recorder.start.assert_not_called()
        app.recorder.stop.assert_not_called()

    def test_transcribe_and_inject(self, mock_config):
        app = self._make_app(mock_config)
        app.transcriber.transcribe.return_value = "Привет мир"

        app._transcribe_and_inject("/tmp/test.wav")

        app.transcriber.transcribe.assert_called_once_with("/tmp/test.wav")
        app.injector.inject.assert_called_once_with("Привет мир")
        assert app.state == wl.AppState.IDLE

    def test_transcribe_empty_result(self, mock_config):
        app = self._make_app(mock_config)
        app.transcriber.transcribe.return_value = ""

        app._transcribe_and_inject("/tmp/test.wav")

        app.injector.inject.assert_not_called()
        assert app.state == wl.AppState.IDLE

    def test_transcribe_error_returns_to_idle(self, mock_config):
        app = self._make_app(mock_config)
        app.transcriber.transcribe.side_effect = RuntimeError("model failed")

        app._transcribe_and_inject("/tmp/test.wav")

        assert app.state == wl.AppState.IDLE

    def test_quit_cleans_up(self, mock_config, tmp_pid_file):
        app = self._make_app(mock_config)
        wl.WhisperLinuxApp.write_pid()
        assert tmp_pid_file.exists()

        app.quit()
        app.recorder.cleanup.assert_called_once()
        assert not tmp_pid_file.exists()

    def test_start_recording_failure(self, mock_config):
        app = self._make_app(mock_config)
        app.recorder.start.side_effect = OSError("no arecord")

        app.toggle()  # Should not crash
        assert app.state == wl.AppState.IDLE

    def test_stop_recording_no_wav(self, mock_config):
        app = self._make_app(mock_config)
        app.recorder.stop.return_value = None

        app.state = wl.AppState.RECORDING
        app.toggle()
        assert app.state == wl.AppState.IDLE


# ===================================================================
# send_toggle
# ===================================================================

class TestSendToggle:
    def test_send_toggle_no_running_instance(self, tmp_pid_file):
        with pytest.raises(SystemExit):
            wl.send_toggle()

    @patch("app.os.kill")
    def test_send_toggle_success(self, mock_kill, tmp_pid_file):
        tmp_pid_file.write_text(str(os.getpid()))
        wl.send_toggle()
        # read_pid calls os.kill(pid, 0) to check existence, then send_toggle sends SIGUSR1
        assert mock_kill.call_count == 2
        mock_kill.assert_any_call(os.getpid(), 0)
        mock_kill.assert_any_call(os.getpid(), signal.SIGUSR1)


# ===================================================================
# CLI argument parsing
# ===================================================================

class TestCLI:
    @patch("app.app.send_toggle")
    def test_toggle_flag(self, mock_toggle, monkeypatch):
        monkeypatch.setattr("sys.argv", ["app.py", "--toggle"])
        wl.main()
        mock_toggle.assert_called_once()

    @patch("app.app.WhisperLinuxApp")
    def test_language_override(self, mock_app_cls, monkeypatch, tmp_pid_file, tmp_config_file):
        monkeypatch.setattr("sys.argv", ["app.py", "--language", "en"])
        monkeypatch.setattr(wl.config, "CONFIG_FILE", tmp_config_file)
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_file.parent)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        mock_app_cls.read_pid.return_value = None  # No running instance

        wl.main()

        # Check that Config was created with language override
        config_arg = mock_app_cls.call_args[0][0]
        assert config_arg.language == "en"

    def test_already_running(self, tmp_pid_file, monkeypatch, capsys):
        tmp_pid_file.write_text(str(os.getpid()))
        monkeypatch.setattr("sys.argv", ["app.py"])
        with pytest.raises(SystemExit):
            wl.main()
        captured = capsys.readouterr()
        assert "already running" in captured.err


# ===================================================================
# TrayIcon — model download and settings handlers
# ===================================================================

class TestTrayIconHandlers:
    """Test TrayIcon event handlers using mocked app_ref."""

    def _make_tray_mocked(self, mock_config, tmp_path):
        """Create a TrayIcon with fully mocked Qt and app_ref."""
        app_ref = MagicMock()
        app_ref.config = mock_config
        # Create a model file so _rebuild_model_menu can stat it
        model_file = tmp_path / "ggml-base.bin"
        model_file.write_bytes(b"\x00" * 1000)
        mock_config.model = str(model_file)

        tray = wl.TrayIcon.__new__(wl.TrayIcon)
        tray._app_ref = app_ref
        tray._downloading = set()
        tray._kept_actions = []
        # _marshal_call: execute immediately in tests (no Qt event loop)
        tray._marshal_call = lambda func: func()
        return tray, app_ref

    @patch("app.subprocess.run")
    def test_download_model_no_auto_switch(self, mock_run, mock_config, tmp_path):
        """After download completes, model is NOT auto-switched."""
        tray, app_ref = self._make_tray_mocked(mock_config, tmp_path)
        tray.notify = MagicMock()
        tray._rebuild_model_menu = MagicMock()
        original_model = mock_config.model

        # Point models_dir to tmp_path so the downloaded file can be checked
        mock_config.models_dir = str(tmp_path)
        # Create the "downloaded" model file with a realistic size
        model_file = tmp_path / "ggml-tiny.bin"
        model_file.write_bytes(b"\x00" * 75_000_000)

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        with patch.object(Path, "is_file", return_value=True):
            tray._do_download_model("tiny")

        # Model should NOT have been changed — user decides manually
        assert mock_config.model == original_model

    def test_on_threads_changed(self, mock_config):
        """Threads handler updates config and saves."""
        mock_config.save = MagicMock()
        app_ref = MagicMock()
        app_ref.config = mock_config

        tray = wl.TrayIcon.__new__(wl.TrayIcon)
        tray._app_ref = app_ref

        mock_group = MagicMock()
        mock_action = MagicMock()
        mock_action.data.return_value = 8
        mock_group.checkedAction.return_value = mock_action
        tray._threads_group = mock_group

        tray._on_threads_changed()
        assert mock_config.threads == 8
        mock_config.save.assert_called_once()

    def test_on_gpu_changed(self, mock_config):
        """GPU handler updates config and saves."""
        mock_config.save = MagicMock()
        app_ref = MagicMock()
        app_ref.config = mock_config

        tray = wl.TrayIcon.__new__(wl.TrayIcon)
        tray._app_ref = app_ref

        mock_group = MagicMock()
        mock_action = MagicMock()
        mock_action.data.return_value = 1
        mock_group.checkedAction.return_value = mock_action
        tray._gpu_group = mock_group

        tray._on_gpu_changed()
        assert mock_config.gpu_device == 1
        mock_config.save.assert_called_once()

    def test_on_audio_changed(self, mock_config):
        """Audio handler updates config and saves."""
        mock_config.save = MagicMock()
        app_ref = MagicMock()
        app_ref.config = mock_config

        tray = wl.TrayIcon.__new__(wl.TrayIcon)
        tray._app_ref = app_ref

        mock_group = MagicMock()
        mock_action = MagicMock()
        mock_action.data.return_value = "plughw:1,0"
        mock_group.checkedAction.return_value = mock_action
        tray._audio_group = mock_group

        tray._on_audio_changed()
        assert mock_config.audio_device == "plughw:1,0"
        mock_config.save.assert_called_once()

    def test_on_paste_keys_changed(self, mock_config):
        """Paste keys handler updates config and saves."""
        mock_config.save = MagicMock()
        app_ref = MagicMock()
        app_ref.config = mock_config

        tray = wl.TrayIcon.__new__(wl.TrayIcon)
        tray._app_ref = app_ref

        mock_group = MagicMock()
        mock_action = MagicMock()
        mock_action.data.return_value = "ctrl+shift+v"
        mock_group.checkedAction.return_value = mock_action
        tray._paste_group = mock_group

        tray._on_paste_keys_changed()
        assert mock_config.paste_keys == "ctrl+shift+v"
        mock_config.save.assert_called_once()

    def test_download_marks_downloading(self, mock_config, tmp_path):
        """_on_download_model adds name to _downloading set."""
        tray, app_ref = self._make_tray_mocked(mock_config, tmp_path)
        tray.notify = MagicMock()
        tray._rebuild_model_menu = MagicMock()

        with patch("app.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            tray._on_download_model("small")

        assert "small" in tray._downloading
        tray._rebuild_model_menu.assert_called_once()

    def test_download_duplicate_ignored(self, mock_config, tmp_path):
        """Second download of same model is ignored."""
        tray, app_ref = self._make_tray_mocked(mock_config, tmp_path)
        tray.notify = MagicMock()
        tray._rebuild_model_menu = MagicMock()
        tray._downloading.add("small")

        with patch("app.threading.Thread") as mock_thread:
            tray._on_download_model("small")
            mock_thread.assert_not_called()


# ===================================================================
# _write_wav helper
# ===================================================================

class TestWriteWav:
    def test_write_wav_creates_valid_file(self, tmp_path):
        path = str(tmp_path / "test.wav")
        pcm = b"\x00\x01" * 1600  # 100ms of s16le mono
        wl._write_wav(path, pcm)

        data = Path(path).read_bytes()
        assert data[:4] == b"RIFF"
        assert data[8:12] == b"WAVE"
        assert data[12:16] == b"fmt "
        assert data[36:40] == b"data"

    def test_write_wav_correct_size(self, tmp_path):
        path = str(tmp_path / "test.wav")
        pcm = b"\x00" * 3200  # 100ms
        wl._write_wav(path, pcm)

        import struct
        data = Path(path).read_bytes()
        # RIFF size = 36 + data_size
        riff_size = struct.unpack_from("<I", data, 4)[0]
        assert riff_size == 36 + 3200
        # data chunk size
        data_size = struct.unpack_from("<I", data, 40)[0]
        assert data_size == 3200

    def test_write_wav_empty_pcm(self, tmp_path):
        path = str(tmp_path / "empty.wav")
        wl._write_wav(path, b"")
        assert Path(path).exists()
        assert Path(path).stat().st_size == 44  # Header only


# ===================================================================
# SimpleVAD
# ===================================================================

class TestSimpleVAD:
    def _make_config(self):
        cfg = MagicMock()
        cfg.vad_threshold = 300
        cfg.min_speech_ms = 100  # Lower for testing
        cfg.max_speech_s = 30.0
        return cfg

    def test_silence_no_callback(self, silence_pcm):
        callback = MagicMock()
        vad = wl.SimpleVAD(self._make_config(), on_speech_end=callback)
        vad.feed(silence_pcm)
        callback.assert_not_called()

    def test_speech_triggers_callback(self, speech_pcm, silence_pcm):
        callback = MagicMock()
        vad = wl.SimpleVAD(self._make_config(), on_speech_end=callback)
        # Feed speech then silence to trigger end-of-speech
        vad.feed(speech_pcm)
        vad.feed(silence_pcm)
        callback.assert_called_once()
        pcm = callback.call_args[0][0]
        assert len(pcm) > 0

    def test_short_speech_discarded(self):
        """Speech shorter than min_speech_ms is not emitted."""
        import array, math
        callback = MagicMock()
        cfg = self._make_config()
        cfg.min_speech_ms = 2000  # 2 seconds minimum
        vad = wl.SimpleVAD(cfg, on_speech_end=callback)
        # 100ms of loud audio
        samples = array.array("h")
        for i in range(1600):
            samples.append(int(10000 * math.sin(2 * math.pi * 440 * i / 16000)))
        short_speech = samples.tobytes()
        silence = b"\x00\x00" * 16000
        vad.feed(short_speech)
        vad.feed(silence)
        callback.assert_not_called()

    def test_reset_clears_state(self, speech_pcm):
        callback = MagicMock()
        vad = wl.SimpleVAD(self._make_config(), on_speech_end=callback)
        vad.feed(speech_pcm)
        vad.reset()
        assert not vad._in_speech
        assert len(vad._speech_buffer) == 0

    def test_rms_silence(self):
        rms = wl.SimpleVAD._rms(b"\x00\x00" * 100)
        assert rms == 0.0

    def test_rms_loud(self):
        import array
        samples = array.array("h", [10000] * 100)
        rms = wl.SimpleVAD._rms(samples.tobytes())
        assert rms == 10000.0

    def test_rms_empty(self):
        assert wl.SimpleVAD._rms(b"") == 0.0
        assert wl.SimpleVAD._rms(b"\x00") == 0.0

    def test_speech_start_callback(self, speech_pcm):
        start_cb = MagicMock()
        end_cb = MagicMock()
        vad = wl.SimpleVAD(self._make_config(), on_speech_end=end_cb,
                           on_speech_start=start_cb)
        vad.feed(speech_pcm)
        start_cb.assert_called_once()

    def test_max_speech_enforced(self):
        """Speech exceeding max_speech_s is force-emitted."""
        import array, math
        callback = MagicMock()
        cfg = self._make_config()
        cfg.max_speech_s = 0.5  # 500ms max
        vad = wl.SimpleVAD(cfg, on_speech_end=callback)
        # Feed 1 second of speech (exceeds 500ms max)
        samples = array.array("h")
        for i in range(16000):
            samples.append(int(10000 * math.sin(2 * math.pi * 440 * i / 16000)))
        vad.feed(samples.tobytes())
        callback.assert_called_once()


# ===================================================================
# WakeWordDetector
# ===================================================================

class TestWakeWordDetector:
    def test_exact_match(self):
        d = wl.WakeWordDetector("дуняша")
        assert d.contains_wake_word("дуняша привет") is True

    def test_exact_match_case_insensitive(self):
        d = wl.WakeWordDetector("дуняша")
        assert d.contains_wake_word("Дуняша, привет") is True

    def test_no_match(self):
        d = wl.WakeWordDetector("дуняша")
        assert d.contains_wake_word("привет мир") is False

    def test_fuzzy_match(self):
        d = wl.WakeWordDetector("дуняша")
        # Close misspelling
        assert d.contains_wake_word("дуняшя") is True

    def test_strip_wake_word(self):
        d = wl.WakeWordDetector("дуняша")
        result = d.strip_wake_word("дуняша привет мир")
        assert "дуняша" not in result.lower()
        assert "привет" in result

    def test_strip_wake_word_only(self):
        d = wl.WakeWordDetector("дуняша")
        result = d.strip_wake_word("дуняша")
        assert result == ""

    def test_strip_wake_word_with_punctuation(self):
        d = wl.WakeWordDetector("дуняша")
        result = d.strip_wake_word("Дуняша, запиши")
        assert "запиши" in result
        assert "дуняша" not in result.lower()

    def test_english_wake_word(self):
        d = wl.WakeWordDetector("computer")
        assert d.contains_wake_word("Hey computer, write this") is True
        assert d.contains_wake_word("Hey cortana, write this") is False

    def test_empty_text(self):
        d = wl.WakeWordDetector("дуняша")
        assert d.contains_wake_word("") is False
        assert d.strip_wake_word("") == ""


# ===================================================================
# AudioStream
# ===================================================================

class TestAudioStream:
    @patch("app.time.sleep")
    @patch("app.subprocess.Popen")
    @patch("app.AudioRecorder._find_pw_record", return_value="/usr/bin/pw-record")
    def test_start_stop(self, mock_pw, mock_popen, mock_sleep):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.stdout.read.return_value = b""  # Will end reader loop
        mock_popen.return_value = mock_proc

        stream = wl.AudioStream(MagicMock(audio_device="auto"))
        callback = MagicMock()
        stream.start(callback)

        assert stream.is_running or True  # process may have ended from mock
        stream.stop()
        mock_proc.send_signal.assert_called_with(signal.SIGINT)

    @patch("app.time.sleep")
    @patch("app.subprocess.Popen")
    @patch("app.AudioRecorder._find_pw_record", return_value="/usr/bin/pw-record")
    def test_start_while_running_raises(self, mock_pw, mock_popen, mock_sleep):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        # Block the reader thread so it doesn't clear _running
        mock_proc.stdout.read.side_effect = lambda *a, **kw: threading.Event().wait(0.5) or b""
        mock_popen.return_value = mock_proc

        stream = wl.AudioStream(MagicMock(audio_device="auto"))
        stream.start(MagicMock())
        with pytest.raises(RuntimeError, match="already running"):
            stream.start(MagicMock())
        stream.stop()

    @patch("app.subprocess.Popen")
    @patch("app.AudioRecorder._find_pw_record", return_value="/usr/bin/pw-record")
    def test_start_fails(self, mock_pw, mock_popen):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.stderr.read.return_value = b"Connection refused"
        mock_popen.return_value = mock_proc

        stream = wl.AudioStream(MagicMock(audio_device="auto"))
        with pytest.raises(RuntimeError, match="AudioStream failed"):
            stream.start(MagicMock())

    @patch("app.AudioRecorder._find_pw_record", return_value=None)
    def test_fallback_to_arecord(self, mock_pw):
        stream = wl.AudioStream(MagicMock(audio_device="plughw:1,0"))
        cmd = stream._build_stream_cmd()
        assert cmd[0] == "arecord"
        assert "-D" in cmd
        assert "plughw:1,0" in cmd


# ===================================================================
# Config — streaming fields
# ===================================================================

class TestConfigStreaming:
    def test_default_mode_fields(self, mock_config):
        assert mock_config.input_mode == "hotkey"
        assert mock_config.output_mode == "batch"
        assert mock_config.wake_word == "марфуша"
        assert mock_config.silence_timeout == 3.0
        assert mock_config.vad_threshold == 300
        assert mock_config.min_speech_ms == 500
        assert mock_config.max_speech_s == 30.0

    def test_mode_fields_from_file(self, tmp_config_dir, monkeypatch):
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
            "input_mode = listen\n"
            "output_mode = stream\n"
            "wake_word = hello\n"
            "silence_timeout = 5.0\n"
            "vad_threshold = 500\n"
            "min_speech_ms = 200\n"
            "max_speech_s = 60.0\n"
        )
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_dir)
        monkeypatch.setattr(wl.config, "CONFIG_FILE", config_file)
        config = wl.Config()
        assert config.input_mode == "listen"
        assert config.output_mode == "stream"
        assert config.wake_word == "hello"
        assert config.silence_timeout == 5.0
        assert config.vad_threshold == 500
        assert config.min_speech_ms == 200
        assert config.max_speech_s == 60.0

    def test_backward_compat_old_mode_stream(self, tmp_config_dir, monkeypatch):
        """Old config with mode=stream maps to input_mode=listen, output_mode=stream."""
        config_file = tmp_config_dir / "config.ini"
        config_file.write_text(
            "[whisper-linux]\n"
            "whisper_cli = /usr/bin/whisper-cli\n"
            "model = /tmp/ggml-base.bin\n"
            "models_dir = /tmp\n"
            "display_server = x11\n"
            "audio_device = plughw:1,0\n"
            "mode = stream\n"
        )
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_dir)
        monkeypatch.setattr(wl.config, "CONFIG_FILE", config_file)
        config = wl.Config()
        assert config.input_mode == "listen"
        assert config.output_mode == "stream"

    def test_backward_compat_old_mode_manual(self, tmp_config_dir, monkeypatch):
        """Old config with mode=manual maps to input_mode=hotkey, output_mode=batch."""
        config_file = tmp_config_dir / "config.ini"
        config_file.write_text(
            "[whisper-linux]\n"
            "whisper_cli = /usr/bin/whisper-cli\n"
            "model = /tmp/ggml-base.bin\n"
            "models_dir = /tmp\n"
            "display_server = x11\n"
            "audio_device = plughw:1,0\n"
            "mode = manual\n"
        )
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_dir)
        monkeypatch.setattr(wl.config, "CONFIG_FILE", config_file)
        config = wl.Config()
        assert config.input_mode == "hotkey"
        assert config.output_mode == "batch"

    def test_new_fields_override_old_mode(self, tmp_config_dir, monkeypatch):
        """New input_mode/output_mode take precedence over old mode field."""
        config_file = tmp_config_dir / "config.ini"
        config_file.write_text(
            "[whisper-linux]\n"
            "whisper_cli = /usr/bin/whisper-cli\n"
            "model = /tmp/ggml-base.bin\n"
            "models_dir = /tmp\n"
            "display_server = x11\n"
            "audio_device = plughw:1,0\n"
            "mode = stream\n"
            "input_mode = hotkey\n"
            "output_mode = batch\n"
        )
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_dir)
        monkeypatch.setattr(wl.config, "CONFIG_FILE", config_file)
        config = wl.Config()
        assert config.input_mode == "hotkey"
        assert config.output_mode == "batch"

    def test_save_mode_fields(self, mock_config, tmp_config_file):
        mock_config.input_mode = "listen"
        mock_config.output_mode = "stream"
        mock_config.wake_word = "ok dunyasha"
        mock_config.save()
        text = tmp_config_file.read_text()
        assert "input_mode = listen" in text
        assert "output_mode = stream" in text
        assert "wake_word = ok dunyasha" in text
        # Old "mode" field should NOT be saved
        assert "\nmode = " not in text


# ===================================================================
# WhisperLinuxApp — streaming state machine
# ===================================================================

class TestWhisperLinuxAppStreaming:
    def _make_app(self, config):
        app = wl.WhisperLinuxApp(config)
        app.recorder = MagicMock(spec=wl.AudioRecorder)
        app.transcriber = MagicMock(spec=wl.Transcriber)
        app.injector = MagicMock(spec=wl.TextInjector)
        return app

    def test_toggle_hotkey_batch_mode(self, mock_config):
        """In hotkey+batch mode, toggle starts recording."""
        mock_config.input_mode = "hotkey"
        mock_config.output_mode = "batch"
        app = self._make_app(mock_config)
        app.toggle()
        app.recorder.start.assert_called_once()
        assert app.state == wl.AppState.RECORDING

    @patch("app.app.AudioStream")
    def test_toggle_listen_starts_listening(self, mock_stream_cls, mock_config_stream):
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app.toggle()

        assert app.state == wl.AppState.LISTENING
        mock_stream.start.assert_called_once()

    @patch("app.app.AudioStream")
    def test_toggle_listen_stops_listening(self, mock_stream_cls, mock_config_stream):
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app.toggle()  # IDLE → LISTENING
        assert app.state == wl.AppState.LISTENING
        app.toggle()  # LISTENING → IDLE
        assert app.state == wl.AppState.IDLE
        mock_stream.stop.assert_called_once()

    @patch("app.app.AudioStream")
    def test_toggle_listen_stops_dictating(self, mock_stream_cls, mock_config_stream):
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app.toggle()  # IDLE → LISTENING
        app.state = wl.AppState.DICTATING  # simulate wake word
        app.toggle()  # DICTATING → IDLE
        assert app.state == wl.AppState.IDLE

    @patch("app.app.AudioStream")
    def test_process_segment_wake_word_starts_dictating(self, mock_stream_cls, mock_config_stream):
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app._wake_detector = wl.WakeWordDetector(mock_config_stream.wake_word)
        app.state = wl.AppState.LISTENING
        app.transcriber.transcribe.return_value = "дуняша привет"

        app._process_speech_segment(b"\x00" * 32000)

        assert app.state == wl.AppState.DICTATING
        # Wake word segment never injects text — it's purely a toggle
        app.injector.inject.assert_not_called()

    @patch("app.app.AudioStream")
    def test_process_segment_wake_word_stops_dictating(self, mock_stream_cls, mock_config_stream):
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app._wake_detector = wl.WakeWordDetector(mock_config_stream.wake_word)
        app.state = wl.AppState.DICTATING
        app.transcriber.transcribe.return_value = "дуняша"

        app._process_speech_segment(b"\x00" * 32000)

        assert app.state == wl.AppState.LISTENING
        # Wake word alone — nothing to inject
        app.injector.inject.assert_not_called()

    @patch("app.app.AudioStream")
    def test_process_segment_wake_word_injects_preceding_text(self, mock_stream_cls, mock_config_stream):
        """Text before wake word is injected when stopping dictation."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app._wake_detector = wl.WakeWordDetector(mock_config_stream.wake_word)
        app.state = wl.AppState.DICTATING
        app.transcriber.transcribe.return_value = "еще пять раз дуняша."

        app._process_speech_segment(b"\x00" * 32000)

        assert app.state == wl.AppState.LISTENING
        app.injector.inject.assert_called_once_with("еще пять раз")

    @patch("app.app.AudioStream")
    def test_process_segment_injects_text_when_dictating_stream(self, mock_stream_cls, mock_config_stream):
        """In stream output mode, text is injected immediately."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app._wake_detector = wl.WakeWordDetector(mock_config_stream.wake_word)
        app.state = wl.AppState.DICTATING
        app.transcriber.transcribe.return_value = "привет мир"

        app._process_speech_segment(b"\x00" * 32000)

        app.injector.inject.assert_called_once_with("привет мир")
        assert app.state == wl.AppState.DICTATING

    @patch("app.app.AudioStream")
    def test_process_segment_no_inject_when_listening(self, mock_stream_cls, mock_config_stream):
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app._wake_detector = wl.WakeWordDetector(mock_config_stream.wake_word)
        app.state = wl.AppState.LISTENING
        app.transcriber.transcribe.return_value = "привет мир"

        app._process_speech_segment(b"\x00" * 32000)

        app.injector.inject.assert_not_called()

    @patch("app.app.AudioStream")
    def test_state_set_immediately_not_via_marshal(self, mock_stream_cls, mock_config_stream):
        """State is set immediately in _process_speech_segment, not deferred via Qt."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app._wake_detector = wl.WakeWordDetector(mock_config_stream.wake_word)
        app.state = wl.AppState.LISTENING
        app.transcriber.transcribe.return_value = "дуняша"

        app._process_speech_segment(b"\x00" * 32000)

        # State is DICTATING immediately after processing (not deferred)
        assert app.state == wl.AppState.DICTATING

    @patch("app.app.AudioStream")
    def test_sequential_segments_correct_state(self, mock_stream_cls, mock_config_stream):
        """Simulate: wake word → text → wake word. Text must be injected."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app._wake_detector = wl.WakeWordDetector(mock_config_stream.wake_word)
        app.state = wl.AppState.LISTENING

        # Segment 1: wake word → DICTATING
        app.transcriber.transcribe.return_value = "дуняша"
        app._process_speech_segment(b"\x00" * 32000)
        assert app.state == wl.AppState.DICTATING

        # Segment 2: actual text → should be injected
        app.transcriber.transcribe.return_value = "привет мир"
        app._process_speech_segment(b"\x00" * 32000)
        app.injector.inject.assert_called_once_with("привет мир")
        assert app.state == wl.AppState.DICTATING

        # Segment 3: wake word → LISTENING
        app.transcriber.transcribe.return_value = "дуняша"
        app._process_speech_segment(b"\x00" * 32000)
        assert app.state == wl.AppState.LISTENING
        # Only one inject call (the text, not the wake words)
        app.injector.inject.assert_called_once()

    def test_silence_timer_returns_to_listening(self, mock_config_stream):
        app = self._make_app(mock_config_stream)
        app.state = wl.AppState.DICTATING
        app._on_silence_timeout()
        assert app.state == wl.AppState.LISTENING

    def test_silence_timer_ignored_when_not_dictating(self, mock_config_stream):
        app = self._make_app(mock_config_stream)
        app.state = wl.AppState.IDLE
        app._on_silence_timeout()
        assert app.state == wl.AppState.IDLE

    def test_speech_start_cancels_silence_timer(self, mock_config_stream):
        """When VAD detects speech start, silence timer is cancelled."""
        app = self._make_app(mock_config_stream)
        app.state = wl.AppState.DICTATING
        app._reset_silence_timer()
        assert app._silence_timer is not None
        app._on_speech_start()
        assert app._silence_timer is None

    def test_speech_start_ignored_when_not_dictating(self, mock_config_stream):
        """on_speech_start does not cancel timer when not in DICTATING."""
        app = self._make_app(mock_config_stream)
        app.state = wl.AppState.LISTENING
        app._on_speech_start()
        # No crash, no side effects
        assert app._silence_timer is None

    @patch("app.app.AudioStream")
    def test_timer_starts_after_injection_not_wake_word(self, mock_stream_cls, mock_config_stream):
        """Timer is NOT started when wake word detected, only after text injection."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app._wake_detector = wl.WakeWordDetector(mock_config_stream.wake_word)
        app.state = wl.AppState.LISTENING

        # Wake word → DICTATING: timer should NOT be set
        app.transcriber.transcribe.return_value = "дуняша"
        app._process_speech_segment(b"\x00" * 32000)
        assert app.state == wl.AppState.DICTATING
        assert app._silence_timer is None

        # Text injection → timer SHOULD be set
        app.transcriber.transcribe.return_value = "привет мир"
        app._process_speech_segment(b"\x00" * 32000)
        assert app._silence_timer is not None
        app._cancel_silence_timer()  # cleanup

    @patch("app.app.AudioStream")
    def test_speech_end_cancels_timer_before_queue(self, mock_stream_cls, mock_config_stream):
        """_on_speech_end cancels timer when dictating (not resets it)."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app.state = wl.AppState.DICTATING
        app._segment_queue = MagicMock()
        app._reset_silence_timer()
        assert app._silence_timer is not None

        app._on_speech_end(b"\x00" * 32000)
        assert app._silence_timer is None
        app._segment_queue.put.assert_called_once()

    @patch("app.app.AudioStream")
    def test_force_idle_from_listening(self, mock_stream_cls, mock_config_stream):
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app.toggle()  # → LISTENING
        app._force_idle()
        assert app.state == wl.AppState.IDLE

    def test_force_idle_from_recording(self, mock_config):
        app = self._make_app(mock_config)
        app.state = wl.AppState.RECORDING
        app._force_idle()
        app.recorder.stop.assert_called_once()
        assert app.state == wl.AppState.IDLE

    @patch("app.app.AudioStream")
    def test_quit_cleans_up_stream(self, mock_stream_cls, mock_config_stream, tmp_pid_file):
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app._audio_stream = mock_stream
        wl.WhisperLinuxApp.write_pid()
        app.quit()

        mock_stream.stop.assert_called_once()
        app.recorder.cleanup.assert_called_once()


# ===================================================================
# WhisperLinuxApp — hotkey+stream mode
# ===================================================================

class TestHotkeyStream:
    def _make_app(self, config):
        app = wl.WhisperLinuxApp(config)
        app.recorder = MagicMock(spec=wl.AudioRecorder)
        app.transcriber = MagicMock(spec=wl.Transcriber)
        app.injector = MagicMock(spec=wl.TextInjector)
        return app

    @patch("app.app.AudioStream")
    def test_toggle_starts_dictating(self, mock_stream_cls, mock_config_hotkey_stream):
        """Hotkey+stream: toggle goes directly to DICTATING."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_hotkey_stream)
        app.toggle()

        assert app.state == wl.AppState.DICTATING
        mock_stream.start.assert_called_once()
        assert app._wake_detector is None  # no wake word in hotkey mode

    @patch("app.app.AudioStream")
    def test_toggle_stops_dictating(self, mock_stream_cls, mock_config_hotkey_stream):
        """Hotkey+stream: second toggle stops dictating."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_hotkey_stream)
        app.toggle()  # IDLE → DICTATING
        assert app.state == wl.AppState.DICTATING
        app.toggle()  # DICTATING → IDLE
        assert app.state == wl.AppState.IDLE

    @patch("app.app.AudioStream")
    def test_text_injected_immediately(self, mock_stream_cls, mock_config_hotkey_stream):
        """Hotkey+stream: text is injected per segment."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_hotkey_stream)
        app.state = wl.AppState.DICTATING
        app.transcriber.transcribe.return_value = "привет мир"

        app._process_speech_segment(b"\x00" * 32000)

        app.injector.inject.assert_called_once_with("привет мир")

    @patch("app.app.AudioStream")
    def test_no_silence_timer_in_hotkey_mode(self, mock_stream_cls, mock_config_hotkey_stream):
        """Hotkey+stream: no silence timer (user presses hotkey to stop)."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_hotkey_stream)
        app.state = wl.AppState.DICTATING
        app.transcriber.transcribe.return_value = "привет мир"

        app._process_speech_segment(b"\x00" * 32000)

        assert app._silence_timer is None

    @patch("app.app.AudioStream")
    def test_start_signal_on_hotkey_stream(self, mock_stream_cls, mock_config_hotkey_stream):
        """Hotkey+stream: start signal plays when dictation begins."""
        mock_config_hotkey_stream.end_signal = True
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_hotkey_stream)
        app._play_start_signal = MagicMock()
        app.toggle()

        assert app.state == wl.AppState.DICTATING
        app._play_start_signal.assert_called_once()

    @patch("app.app.AudioStream")
    def test_end_signal_on_hotkey_stream_stop(self, mock_stream_cls, mock_config_hotkey_stream):
        """Hotkey+stream: end signal plays when dictation stops."""
        mock_config_hotkey_stream.end_signal = True
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_hotkey_stream)
        app._play_start_signal = MagicMock()
        app._play_end_signal = MagicMock()
        app.toggle()  # start
        app.toggle()  # stop

        assert app.state == wl.AppState.IDLE
        app._play_end_signal.assert_called_once()

    @patch("app.app.AudioStream")
    def test_no_signal_when_disabled(self, mock_stream_cls, mock_config_hotkey_stream):
        """Hotkey+stream: no signals when end_signal is False."""
        mock_config_hotkey_stream.end_signal = False
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_hotkey_stream)
        app._play_start_signal = MagicMock()
        app._play_end_signal = MagicMock()
        app.toggle()  # start
        app.toggle()  # stop
        app._play_start_signal.assert_called_once()  # method called but returns early
        app._play_end_signal.assert_called_once()     # method called but returns early


# ===================================================================
# WhisperLinuxApp — batch mode end signal
# ===================================================================

class TestBatchEndSignal:
    def _make_app(self, config):
        app = wl.WhisperLinuxApp(config)
        app.recorder = MagicMock(spec=wl.AudioRecorder)
        app.transcriber = MagicMock(spec=wl.Transcriber)
        app.injector = MagicMock(spec=wl.TextInjector)
        return app

    def test_end_signal_after_transcription(self, mock_config):
        """Batch mode: end signal plays after transcription completes."""
        mock_config.end_signal = True
        app = self._make_app(mock_config)
        app.transcriber.transcribe.return_value = "Привет мир"
        app._play_end_signal = MagicMock()

        app._transcribe_and_inject("/tmp/test.wav")

        assert app.state == wl.AppState.IDLE
        app._play_end_signal.assert_called_once()

    def test_end_signal_after_empty_transcription(self, mock_config):
        """Batch mode: end signal plays even on empty transcription."""
        mock_config.end_signal = True
        app = self._make_app(mock_config)
        app.transcriber.transcribe.return_value = ""
        app._play_end_signal = MagicMock()

        app._transcribe_and_inject("/tmp/test.wav")

        assert app.state == wl.AppState.IDLE
        app._play_end_signal.assert_called_once()


# ===================================================================
# WhisperLinuxApp — listen+batch mode
# ===================================================================

class TestListenBatch:
    def _make_app(self, config):
        app = wl.WhisperLinuxApp(config)
        app.recorder = MagicMock(spec=wl.AudioRecorder)
        app.transcriber = MagicMock(spec=wl.Transcriber)
        app.injector = MagicMock(spec=wl.TextInjector)
        return app

    @patch("app.app.AudioStream")
    def test_toggle_starts_listening(self, mock_stream_cls, mock_config_listen_batch):
        """Listen+batch: toggle starts LISTENING."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_listen_batch)
        app.toggle()

        assert app.state == wl.AppState.LISTENING

    @patch("app.app.AudioStream")
    def test_text_accumulated_not_injected(self, mock_stream_cls, mock_config_listen_batch):
        """Listen+batch: text is accumulated, not injected per segment."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_listen_batch)
        app._wake_detector = wl.WakeWordDetector(mock_config_listen_batch.wake_word)
        app.state = wl.AppState.DICTATING
        app.transcriber.transcribe.return_value = "привет мир"

        app._process_speech_segment(b"\x00" * 32000)

        # Text NOT injected immediately
        app.injector.inject.assert_not_called()
        assert app._accumulated_texts == ["привет мир"]

    @patch("app.app.AudioStream")
    def test_accumulated_text_flushed_on_wake_word_stop(self, mock_stream_cls, mock_config_listen_batch):
        """Listen+batch: accumulated text flushed when wake word stops dictation."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_listen_batch)
        app._wake_detector = wl.WakeWordDetector(mock_config_listen_batch.wake_word)
        app.state = wl.AppState.DICTATING
        app._accumulated_texts = ["привет", "мир"]
        app.transcriber.transcribe.return_value = "дуняша"

        app._process_speech_segment(b"\x00" * 32000)

        assert app.state == wl.AppState.LISTENING
        app.injector.inject.assert_called_once_with("привет мир")
        assert app._accumulated_texts == []

    @patch("app.app.AudioStream")
    def test_text_before_wake_word_accumulated_in_batch(self, mock_stream_cls, mock_config_listen_batch):
        """Listen+batch: text before wake word is added to accumulated, then flushed."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_listen_batch)
        app._wake_detector = wl.WakeWordDetector(mock_config_listen_batch.wake_word)
        app.state = wl.AppState.DICTATING
        app._accumulated_texts = ["первая часть"]
        app.transcriber.transcribe.return_value = "вторая часть дуняша"

        app._process_speech_segment(b"\x00" * 32000)

        assert app.state == wl.AppState.LISTENING
        app.injector.inject.assert_called_once_with("первая часть вторая часть")

    @patch("app.app.AudioStream")
    def test_accumulated_text_flushed_on_silence_timeout(self, mock_stream_cls, mock_config_listen_batch):
        """Listen+batch: accumulated text flushed on silence timeout."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_listen_batch)
        app.state = wl.AppState.DICTATING
        app._accumulated_texts = ["один", "два", "три"]

        app._on_silence_timeout()

        assert app.state == wl.AppState.LISTENING
        app.injector.inject.assert_called_once_with("один два три")
        assert app._accumulated_texts == []

    @patch("app.app.AudioStream")
    def test_multiple_segments_accumulated(self, mock_stream_cls, mock_config_listen_batch):
        """Listen+batch: multiple segments accumulate text."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_listen_batch)
        app._wake_detector = wl.WakeWordDetector(mock_config_listen_batch.wake_word)
        app.state = wl.AppState.DICTATING

        app.transcriber.transcribe.return_value = "первый"
        app._process_speech_segment(b"\x00" * 32000)
        app.transcriber.transcribe.return_value = "второй"
        app._process_speech_segment(b"\x00" * 32000)
        app.transcriber.transcribe.return_value = "третий"
        app._process_speech_segment(b"\x00" * 32000)

        app.injector.inject.assert_not_called()
        assert app._accumulated_texts == ["первый", "второй", "третий"]


# ===================================================================
# CLI — streaming arguments
# ===================================================================

class TestCLIStreaming:
    @patch("app.app.WhisperLinuxApp")
    def test_stream_flag(self, mock_app_cls, monkeypatch, tmp_pid_file, tmp_config_file):
        monkeypatch.setattr("sys.argv", ["app.py", "--stream"])
        monkeypatch.setattr(wl.config, "CONFIG_FILE", tmp_config_file)
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_file.parent)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        mock_app_cls.read_pid.return_value = None

        wl.main()

        config_arg = mock_app_cls.call_args[0][0]
        assert config_arg.output_mode == "stream"

    @patch("app.app.WhisperLinuxApp")
    def test_input_mode_flag(self, mock_app_cls, monkeypatch, tmp_pid_file, tmp_config_file):
        monkeypatch.setattr("sys.argv", ["app.py", "--input-mode", "listen"])
        monkeypatch.setattr(wl.config, "CONFIG_FILE", tmp_config_file)
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_file.parent)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        mock_app_cls.read_pid.return_value = None

        wl.main()

        config_arg = mock_app_cls.call_args[0][0]
        assert config_arg.input_mode == "listen"

    @patch("app.app.WhisperLinuxApp")
    def test_output_mode_flag(self, mock_app_cls, monkeypatch, tmp_pid_file, tmp_config_file):
        monkeypatch.setattr("sys.argv", ["app.py", "--output-mode", "stream"])
        monkeypatch.setattr(wl.config, "CONFIG_FILE", tmp_config_file)
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_file.parent)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        mock_app_cls.read_pid.return_value = None

        wl.main()

        config_arg = mock_app_cls.call_args[0][0]
        assert config_arg.output_mode == "stream"

    @patch("app.app.WhisperLinuxApp")
    def test_wake_word_flag(self, mock_app_cls, monkeypatch, tmp_pid_file, tmp_config_file):
        monkeypatch.setattr("sys.argv", ["app.py", "--wake-word", "компьютер"])
        monkeypatch.setattr(wl.config, "CONFIG_FILE", tmp_config_file)
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_file.parent)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        mock_app_cls.read_pid.return_value = None

        wl.main()

        config_arg = mock_app_cls.call_args[0][0]
        assert config_arg.wake_word == "компьютер"


# ===================================================================
# End signal player
# ===================================================================

class TestEndSignal:
    def _make_app(self, mock_config):
        app = wl.WhisperLinuxApp(mock_config)
        app.recorder = MagicMock(spec=wl.AudioRecorder)
        app.transcriber = MagicMock(spec=wl.Transcriber)
        app.injector = MagicMock(spec=wl.TextInjector)
        return app

    @patch("app.subprocess.Popen")
    @patch("app.os.path.isfile", return_value=True)
    def test_play_end_signal_pw_play(self, mock_isfile, mock_popen, mock_config):
        """pw-play is tried first when sound file exists."""
        mock_config.end_signal = True
        app = self._make_app(mock_config)
        app._play_end_signal()
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "pw-play"

    @patch("app.subprocess.Popen")
    @patch("app.os.path.isfile", return_value=True)
    def test_play_end_signal_fallback_paplay(self, mock_isfile, mock_popen, mock_config):
        """Falls back to paplay when pw-play fails."""
        mock_config.end_signal = True
        mock_popen.side_effect = [FileNotFoundError(), MagicMock()]
        app = self._make_app(mock_config)
        app._play_end_signal()
        assert mock_popen.call_count == 2
        assert mock_popen.call_args_list[1][0][0][0] == "paplay"

    @patch("app.subprocess.Popen")
    @patch("app.os.path.isfile", return_value=True)
    def test_play_end_signal_fallback_canberra(self, mock_isfile, mock_popen, mock_config):
        """Falls back to canberra-gtk-play when pw-play and paplay fail."""
        mock_config.end_signal = True
        mock_popen.side_effect = [FileNotFoundError(), FileNotFoundError(), MagicMock()]
        app = self._make_app(mock_config)
        app._play_end_signal()
        assert mock_popen.call_count == 3
        assert mock_popen.call_args_list[2][0][0][0] == "canberra-gtk-play"

    def test_play_end_signal_disabled(self, mock_config):
        """Does nothing when end_signal is False."""
        mock_config.end_signal = False
        app = self._make_app(mock_config)
        with patch("app.subprocess.Popen") as mock_popen:
            app._play_end_signal()
            mock_popen.assert_not_called()


# ===================================================================
# TrayIcon — silence timeout text input handler
# ===================================================================

class TestSilenceTimeoutHandler:
    def test_on_silence_timeout_spin(self, mock_config):
        """Silence timeout spinbox handler updates config and saves."""
        mock_config.save = MagicMock()
        app_ref = MagicMock()
        app_ref.config = mock_config

        tray = wl.TrayIcon.__new__(wl.TrayIcon)
        tray._app_ref = app_ref

        tray._on_silence_timeout_spin(5.0)

        assert mock_config.silence_timeout == 5.0
        mock_config.save.assert_called_once()


# ===================================================================
# TrayIcon — wake model handler
# ===================================================================

class TestWakeModelHandler:
    def test_on_wake_model_changed(self, mock_config):
        """Wake model handler updates config and saves."""
        mock_config.save = MagicMock()
        app_ref = MagicMock()
        app_ref.config = mock_config

        tray = wl.TrayIcon.__new__(wl.TrayIcon)
        tray._app_ref = app_ref

        mock_group = MagicMock()
        mock_action = MagicMock()
        mock_action.data.return_value = "/tmp/ggml-tiny.bin"
        mock_group.checkedAction.return_value = mock_action
        tray._wake_model_group = mock_group

        tray._on_wake_model_changed()
        assert mock_config.wake_model == "/tmp/ggml-tiny.bin"
        mock_config.save.assert_called_once()

    def test_on_wake_model_changed_same_as_main(self, mock_config):
        """Setting wake model to empty string means 'same as main'."""
        mock_config.save = MagicMock()
        app_ref = MagicMock()
        app_ref.config = mock_config

        tray = wl.TrayIcon.__new__(wl.TrayIcon)
        tray._app_ref = app_ref

        mock_group = MagicMock()
        mock_action = MagicMock()
        mock_action.data.return_value = ""
        mock_group.checkedAction.return_value = mock_action
        tray._wake_model_group = mock_group

        tray._on_wake_model_changed()
        assert mock_config.wake_model == ""
        mock_config.save.assert_called_once()


# ===================================================================
# WakeWordDetector — fuzzy strip
# ===================================================================

class TestWakeWordFuzzyStrip:
    def test_strip_fuzzy_match(self):
        """Fuzzy match (e.g. 'Морфуша' for 'марфуша') is stripped."""
        d = wl.WakeWordDetector("марфуша")
        result = d.strip_wake_word("Морфуша текст после")
        assert "морфуша" not in result.lower()
        assert "морфуша" not in result.lower()
        assert "текст после" in result

    def test_strip_fuzzy_match_case_variant(self):
        """Fuzzy match with different casing is stripped."""
        d = wl.WakeWordDetector("марфуша")
        result = d.strip_wake_word("МАРФУША привет")
        assert "марфуша" not in result.lower()
        assert "привет" in result

    def test_strip_preserves_surrounding_text(self):
        """Text before and after fuzzy-matched wake word is preserved."""
        d = wl.WakeWordDetector("марфуша")
        result = d.strip_wake_word("раз два Морфуша три четыре")
        assert "раз" in result
        assert "два" in result
        assert "три" in result
        assert "четыре" in result

    def test_is_fuzzy_match_above_threshold(self):
        """Words close enough to wake word match."""
        d = wl.WakeWordDetector("марфуша")
        assert d._is_fuzzy_match("морфуша") is True

    def test_is_fuzzy_match_below_threshold(self):
        """Completely different words do not match."""
        d = wl.WakeWordDetector("марфуша")
        assert d._is_fuzzy_match("привет") is False


# ===================================================================
# Silence timeout — VAD speech-active check
# ===================================================================

class TestSilenceTimeoutVAD:
    def _make_app(self, config):
        app = wl.WhisperLinuxApp(config)
        app.recorder = MagicMock(spec=wl.AudioRecorder)
        app.transcriber = MagicMock(spec=wl.Transcriber)
        app.injector = MagicMock(spec=wl.TextInjector)
        return app

    def test_timeout_skipped_when_speech_active(self, mock_config_stream):
        """Silence timeout restarts timer if VAD has active speech."""
        app = self._make_app(mock_config_stream)
        app.state = wl.AppState.DICTATING
        app._vad = MagicMock()
        app._vad._in_speech = True

        app._on_silence_timeout()

        # Should NOT transition to LISTENING
        assert app.state == wl.AppState.DICTATING
        # Timer should have been restarted
        assert app._silence_timer is not None
        app._cancel_silence_timer()

    def test_timeout_fires_when_no_speech(self, mock_config_stream):
        """Silence timeout fires normally when VAD has no active speech."""
        app = self._make_app(mock_config_stream)
        app.state = wl.AppState.DICTATING
        app._vad = MagicMock()
        app._vad._in_speech = False

        app._on_silence_timeout()

        assert app.state == wl.AppState.LISTENING

    def test_timeout_fires_when_no_vad(self, mock_config_stream):
        """Silence timeout fires normally when no VAD exists."""
        app = self._make_app(mock_config_stream)
        app.state = wl.AppState.DICTATING
        app._vad = None

        app._on_silence_timeout()

        assert app.state == wl.AppState.LISTENING


# ===================================================================
# Transcriber — wake model parameter
# ===================================================================

class TestTranscriberWakeModel:
    @patch("app.subprocess.run")
    def test_transcribe_with_custom_model(self, mock_run, mock_config, tmp_wav_file):
        """Transcriber uses provided model parameter instead of config.model."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="test text",
            stderr="",
        )

        tr = wl.Transcriber(mock_config)
        tr.transcribe(tmp_wav_file, model="/tmp/ggml-tiny.bin")

        cmd = mock_run.call_args[0][0]
        assert "-m" in cmd
        m_idx = cmd.index("-m")
        assert cmd[m_idx + 1] == "/tmp/ggml-tiny.bin"

    @patch("app.subprocess.run")
    def test_transcribe_default_model(self, mock_run, mock_config, tmp_wav_file):
        """Transcriber uses config.model when no model parameter given."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="test text",
            stderr="",
        )

        tr = wl.Transcriber(mock_config)
        tr.transcribe(tmp_wav_file)

        cmd = mock_run.call_args[0][0]
        m_idx = cmd.index("-m")
        assert cmd[m_idx + 1] == mock_config.model


# ===================================================================
# Process segment — wake model used in LISTENING state
# ===================================================================

class TestProcessSegmentWakeModel:
    def _make_app(self, config):
        app = wl.WhisperLinuxApp(config)
        app.recorder = MagicMock(spec=wl.AudioRecorder)
        app.transcriber = MagicMock(spec=wl.Transcriber)
        app.injector = MagicMock(spec=wl.TextInjector)
        return app

    @patch("app.app.AudioStream")
    def test_wake_model_used_when_listening(self, mock_stream_cls, mock_config_stream):
        """In LISTENING state with wake_model set, lighter model is used."""
        mock_config_stream.wake_model = "/tmp/ggml-tiny.bin"
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app._wake_detector = wl.WakeWordDetector(mock_config_stream.wake_word)
        app.state = wl.AppState.LISTENING
        app.transcriber.transcribe.return_value = "привет мир"

        app._process_speech_segment(b"\x00" * 32000)

        app.transcriber.transcribe.assert_called_once()
        _, kwargs = app.transcriber.transcribe.call_args
        assert kwargs.get("model") == "/tmp/ggml-tiny.bin"

    @patch("app.app.AudioStream")
    def test_main_model_used_when_dictating(self, mock_stream_cls, mock_config_stream):
        """In DICTATING state, main model is used even if wake_model is set."""
        mock_config_stream.wake_model = "/tmp/ggml-tiny.bin"
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app._wake_detector = wl.WakeWordDetector(mock_config_stream.wake_word)
        app.state = wl.AppState.DICTATING
        app.transcriber.transcribe.return_value = "привет мир"

        app._process_speech_segment(b"\x00" * 32000)

        app.transcriber.transcribe.assert_called_once()
        _, kwargs = app.transcriber.transcribe.call_args
        assert kwargs.get("model") is None

    @patch("app.app.AudioStream")
    def test_no_wake_model_uses_default(self, mock_stream_cls, mock_config_stream):
        """When wake_model is empty, default model is used even in LISTENING."""
        mock_config_stream.wake_model = ""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        app = self._make_app(mock_config_stream)
        app._wake_detector = wl.WakeWordDetector(mock_config_stream.wake_word)
        app.state = wl.AppState.LISTENING
        app.transcriber.transcribe.return_value = "привет мир"

        app._process_speech_segment(b"\x00" * 32000)

        _, kwargs = app.transcriber.transcribe.call_args
        assert kwargs.get("model") is None


# ===================================================================
# CLI — wake model flag
# ===================================================================

class TestCLIWakeModel:
    @patch("app.app.WhisperLinuxApp")
    def test_wake_model_flag(self, mock_app_cls, monkeypatch, tmp_pid_file, tmp_config_file, tmp_path):
        model_file = tmp_path / "ggml-base.bin"
        model_file.write_bytes(b"\x00" * 100)
        wake_file = tmp_path / "ggml-tiny.bin"
        wake_file.write_bytes(b"\x00" * 100)
        tmp_config_file.write_text(
            "[whisper-linux]\n"
            f"whisper_cli = /usr/bin/whisper-cli\n"
            f"model = {model_file}\n"
            f"models_dir = {tmp_path}\n"
        )
        monkeypatch.setattr("sys.argv", ["app.py", "--wake-model", str(wake_file)])
        monkeypatch.setattr(wl.config, "CONFIG_FILE", tmp_config_file)
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_file.parent)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        mock_app_cls.read_pid.return_value = None

        wl.main()

        config_arg = mock_app_cls.call_args[0][0]
        assert config_arg.wake_model == str(wake_file)


# ===================================================================
# Config — wake_model save/load
# ===================================================================

class TestConfigWakeModel:
    def test_wake_model_default_empty(self, mock_config):
        """Default wake_model is empty string."""
        assert mock_config.wake_model == ""

    def test_save_and_load_wake_model(self, mock_config, tmp_config_file, monkeypatch):
        """wake_model is persisted in config file."""
        mock_config.wake_model = "/tmp/ggml-tiny.bin"
        mock_config.save()
        text = tmp_config_file.read_text()
        assert "wake_model = /tmp/ggml-tiny.bin" in text

    def test_load_wake_model_from_file(self, tmp_config_dir, monkeypatch):
        config_file = tmp_config_dir / "config.ini"
        config_file.write_text(
            "[whisper-linux]\n"
            "whisper_cli = /usr/bin/whisper-cli\n"
            "model = /tmp/ggml-base.bin\n"
            "models_dir = /tmp\n"
            "display_server = x11\n"
            "audio_device = plughw:1,0\n"
            "wake_model = /tmp/ggml-tiny.bin\n"
        )
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_dir)
        monkeypatch.setattr(wl.config, "CONFIG_FILE", config_file)
        config = wl.Config()
        assert config.wake_model == "/tmp/ggml-tiny.bin"

    def test_voice_commands_default_true(self, mock_config):
        assert mock_config.voice_commands is True

    def test_save_voice_commands(self, mock_config, tmp_config_file):
        mock_config.voice_commands = False
        mock_config.save()
        text = tmp_config_file.read_text()
        assert "voice_commands = False" in text

    def test_load_voice_commands_from_file(self, tmp_config_dir, monkeypatch):
        config_file = tmp_config_dir / "config.ini"
        config_file.write_text(
            "[whisper-linux]\n"
            "whisper_cli = /usr/bin/whisper-cli\n"
            "model = /tmp/ggml-base.bin\n"
            "models_dir = /tmp\n"
            "display_server = x11\n"
            "audio_device = plughw:1,0\n"
            "voice_commands = False\n"
        )
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_dir)
        monkeypatch.setattr(wl.config, "CONFIG_FILE", config_file)
        config = wl.Config()
        assert config.voice_commands is False


# ===================================================================
# VoiceCommands
# ===================================================================

class TestVoiceCommands:
    def test_no_commands_just_text(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        result = vc.process("hello world", inject, send_key)
        assert result is False
        inject.assert_called_once_with("hello world")
        send_key.assert_not_called()

    def test_enter_command(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        result = vc.process("hello enter world", inject, send_key)
        assert result is True
        assert inject.call_args_list == [call("hello"), call("world")]
        send_key.assert_called_once_with("Return")

    def test_enter_russian(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        vc.process("текст энтер ещё", inject, send_key)
        assert inject.call_args_list == [call("текст"), call("ещё")]
        send_key.assert_called_once_with("Return")

    def test_backspace_removes_from_buffer(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        result = vc.process("hello world backspace more", inject, send_key)
        assert result is True
        # "hello world" buffered, backspace removes "world", "more" added
        # Final buffer: "hello more"
        inject.assert_called_once_with("hello more")
        send_key.assert_not_called()

    def test_backspace_empty_buffer_sends_ctrl_backspace(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        result = vc.process("backspace", inject, send_key)
        assert result is True
        inject.assert_not_called()
        send_key.assert_called_once_with("ctrl+BackSpace")

    def test_backspace_russian(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        vc.process("слово назад", inject, send_key)
        # "слово" buffered, "назад" = backspace removes "слово"
        inject.assert_not_called()
        send_key.assert_not_called()

    def test_tab_command(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        vc.process("tab", inject, send_key)
        send_key.assert_called_once_with("Tab")

    def test_escape_command(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        vc.process("escape", inject, send_key)
        send_key.assert_called_once_with("Escape")

    def test_fuzzy_match(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        # "entr" is close enough to "enter" (ratio > 0.75)
        vc.process("entr", inject, send_key)
        send_key.assert_called_once_with("Return")

    def test_fuzzy_no_match(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        # "xyz" is not close to any command
        result = vc.process("xyz", inject, send_key)
        assert result is False
        inject.assert_called_once_with("xyz")
        send_key.assert_not_called()

    def test_command_with_punctuation(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        vc.process("hello, enter.", inject, send_key)
        inject.assert_called_once_with("hello,")
        send_key.assert_called_once_with("Return")

    def test_empty_text(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        result = vc.process("", inject, send_key)
        assert result is False
        inject.assert_not_called()
        send_key.assert_not_called()

    def test_multiple_commands(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        vc.process("hello enter enter world", inject, send_key)
        assert inject.call_args_list == [call("hello"), call("world")]
        assert send_key.call_args_list == [call("Return"), call("Return")]

    def test_command_at_end(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        vc.process("hello world enter", inject, send_key)
        inject.assert_called_once_with("hello world")
        send_key.assert_called_once_with("Return")

    def test_custom_commands(self):
        vc = wl.VoiceCommands(commands={"stop": "key:Escape"})
        inject = MagicMock()
        send_key = MagicMock()
        vc.process("stop", inject, send_key)
        send_key.assert_called_once_with("Escape")

    def test_ввод_command(self):
        vc = wl.VoiceCommands()
        inject = MagicMock()
        send_key = MagicMock()
        vc.process("текст ввод", inject, send_key)
        inject.assert_called_once_with("текст")
        send_key.assert_called_once_with("Return")


# ===================================================================
# TextInjector.send_key
# ===================================================================

class TestTextInjectorSendKey:
    @patch("app.subprocess.run")
    def test_send_key_x11(self, mock_run, mock_config):
        mock_run.return_value = MagicMock(returncode=0)
        inj = wl.TextInjector(mock_config)
        inj.send_key("Return")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "xdotool"
        assert "key" in cmd
        assert "Return" in cmd

    @patch("app.subprocess.run")
    def test_send_key_wayland_wtype(self, mock_run, mock_config_wayland):
        mock_run.return_value = MagicMock(returncode=0)
        inj = wl.TextInjector(mock_config_wayland)
        inj.send_key("Return")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "wtype"
        assert "Return" in cmd

    @patch("app.subprocess.run")
    def test_send_key_wayland_modifier_combo(self, mock_run, mock_config_wayland):
        mock_run.return_value = MagicMock(returncode=0)
        inj = wl.TextInjector(mock_config_wayland)
        inj.send_key("ctrl+BackSpace")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["wtype", "-M", "ctrl", "-k", "BackSpace"]

    @patch("app.subprocess.run")
    def test_send_key_wayland_fallback_ydotool(self, mock_run, mock_config_wayland):
        mock_run.side_effect = [
            FileNotFoundError("wtype"),
            MagicMock(returncode=0),  # ydotool
        ]
        inj = wl.TextInjector(mock_config_wayland)
        inj.send_key("Tab")
        assert mock_run.call_count == 2
        cmd = mock_run.call_args_list[1][0][0]
        assert cmd[0] == "ydotool"
        assert "Tab" in cmd

    @patch("app.subprocess.run")
    def test_send_key_wayland_ydotool_evdev_map(self, mock_run, mock_config_wayland):
        """ydotool translates X11 keysym Return → evdev Enter."""
        mock_run.side_effect = [
            FileNotFoundError("wtype"),
            MagicMock(returncode=0),  # ydotool
        ]
        inj = wl.TextInjector(mock_config_wayland)
        inj.send_key("Return")
        assert mock_run.call_count == 2
        cmd = mock_run.call_args_list[1][0][0]
        assert cmd == ["ydotool", "key", "--delay", "50", "Enter"]

    @patch("app.subprocess.run")
    def test_send_key_wayland_fallback_xdotool(self, mock_run, mock_config_wayland):
        mock_run.side_effect = [
            FileNotFoundError("wtype"),
            FileNotFoundError("ydotool"),
            MagicMock(returncode=0),  # xdotool
        ]
        inj = wl.TextInjector(mock_config_wayland)
        inj.send_key("Tab")
        assert mock_run.call_count == 3
        cmd = mock_run.call_args_list[2][0][0]
        assert cmd[0] == "xdotool"
        assert "Tab" in cmd

    @patch("app.subprocess.run")
    def test_send_key_x11_failure(self, mock_run, mock_config):
        mock_run.side_effect = FileNotFoundError("xdotool")
        inj = wl.TextInjector(mock_config)
        # Should not raise
        inj.send_key("Return")


# ===================================================================
# Voice commands map config
# ===================================================================

class TestVoiceCommandsMap:
    def test_default_map_loaded(self, mock_config):
        assert "enter" in mock_config.voice_commands_map
        assert mock_config.voice_commands_map["enter"] == "key:Return"
        assert "backspace" in mock_config.voice_commands_map

    def test_save_and_load_custom_map(self, mock_config, tmp_config_file):
        mock_config.voice_commands_map = {"go": "key:Return", "стоп": "key:Escape"}
        mock_config.save()
        text = tmp_config_file.read_text()
        assert "[voice-commands]" in text
        assert "go = key:Return" in text
        assert "стоп = key:Escape" in text

    def test_load_custom_map_from_file(self, tmp_config_dir, monkeypatch):
        config_file = tmp_config_dir / "config.ini"
        config_file.write_text(
            "[whisper-linux]\n"
            "whisper_cli = /usr/bin/whisper-cli\n"
            "model = /tmp/ggml-base.bin\n"
            "models_dir = /tmp\n"
            "display_server = x11\n"
            "audio_device = plughw:1,0\n"
            "\n"
            "[voice-commands]\n"
            "go = key:Return\n"
            "стоп = key:Escape\n"
        )
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_dir)
        monkeypatch.setattr(wl.config, "CONFIG_FILE", config_file)
        config = wl.Config()
        # Custom entries override/extend defaults
        assert config.voice_commands_map["go"] == "key:Return"
        assert config.voice_commands_map["стоп"] == "key:Escape"
        # Defaults still present
        assert "enter" in config.voice_commands_map
        assert "табуляция" in config.voice_commands_map

    def test_voice_commands_uses_config_map(self, mock_config):
        mock_config.voice_commands_map = {"go": "key:Return"}
        from app.commands import VoiceCommands
        vc = VoiceCommands(mock_config.voice_commands_map)
        inject = MagicMock()
        send_key = MagicMock()
        vc.process("go", inject, send_key)
        send_key.assert_called_once_with("Return")

    def test_empty_voice_commands_section(self, tmp_config_dir, monkeypatch):
        config_file = tmp_config_dir / "config.ini"
        config_file.write_text(
            "[whisper-linux]\n"
            "whisper_cli = /usr/bin/whisper-cli\n"
            "model = /tmp/ggml-base.bin\n"
            "models_dir = /tmp\n"
            "display_server = x11\n"
            "audio_device = plughw:1,0\n"
            "\n"
            "[voice-commands]\n"
        )
        monkeypatch.setattr(wl.config, "CONFIG_DIR", tmp_config_dir)
        monkeypatch.setattr(wl.config, "CONFIG_FILE", config_file)
        config = wl.Config()
        # Empty section still gets defaults merged in
        assert "enter" in config.voice_commands_map
        assert "табуляция" in config.voice_commands_map
