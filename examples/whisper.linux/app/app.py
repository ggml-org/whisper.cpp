"""Main application logic, state machine, and CLI for whisper.linux."""

import argparse
import logging
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from . import config as _cfg
from .config import Config, AppState, log
from .audio import AudioRecorder, AudioStream, SimpleVAD, _write_wav
from .transcriber import Transcriber, WakeWordDetector
from .injector import TextInjector
from .commands import VoiceCommands
from .tray import TrayIcon


class WhisperLinuxApp:
    """Main application: state machine, threading, PID file, signal handling."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.state = AppState.IDLE
        self.recorder = AudioRecorder(self.config)
        self.transcriber = Transcriber(self.config)
        self.injector = TextInjector(self.config)
        self._tray: Optional[TrayIcon] = None
        self._qt_app = None
        self._prev_window_id: Optional[str] = None
        self._audio_stream: Optional[AudioStream] = None
        self._vad: Optional[SimpleVAD] = None
        self._wake_detector: Optional[WakeWordDetector] = None
        self._silence_timer: Optional[threading.Timer] = None
        self._segment_queue: Optional[queue.Queue] = None
        self._segment_worker: Optional[threading.Thread] = None
        self._accumulated_texts: list = []
        self._voice_commands = VoiceCommands(self.config.voice_commands_map)
        self._mute_until: float = 0

    def _inject_text(self, text: str):
        """Inject text, processing voice commands if enabled."""
        if self.config.voice_commands:
            self._voice_commands.process(
                text, self.injector.inject, self.injector.send_key,
            )
        else:
            self.injector.inject(text)

    # -- PID file management --

    @staticmethod
    def write_pid():
        _cfg.PID_FILE.write_text(str(os.getpid()))

    @staticmethod
    def read_pid() -> Optional[int]:
        if _cfg.PID_FILE.exists():
            try:
                pid = int(_cfg.PID_FILE.read_text().strip())
                os.kill(pid, 0)
                return pid
            except (ValueError, ProcessLookupError, PermissionError):
                return None
        return None

    @staticmethod
    def remove_pid():
        _cfg.PID_FILE.unlink(missing_ok=True)

    # -- Signal handling --

    def _setup_signals(self):
        signal.signal(signal.SIGUSR1, self._on_sigusr1)
        signal.signal(signal.SIGTERM, self._on_sigterm)
        signal.signal(signal.SIGINT, self._on_sigterm)

    def _on_sigusr1(self, signum, frame):
        log.info("Received SIGUSR1 \u2014 toggling")
        if self._qt_app:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, self.toggle)

    def _on_sigterm(self, signum, frame):
        log.info("Received %s \u2014 quitting", signal.Signals(signum).name)
        self.quit()

    # -- State machine --

    def toggle(self):
        log.info("Toggle: state=%s, input=%s, output=%s",
                 self.state.value, self.config.input_mode, self.config.output_mode)
        if self.config.input_mode == "listen":
            self._toggle_listen()
        else:
            self._toggle_hotkey()

    def _toggle_hotkey(self):
        if self.state == AppState.IDLE:
            if self.config.output_mode == "batch":
                self._start_recording()
            else:
                self._start_hotkey_stream()
        elif self.state == AppState.RECORDING:
            self._stop_recording()
        elif self.state == AppState.DICTATING:
            self._stop_hotkey_stream()

    def _toggle_listen(self):
        if self.state == AppState.IDLE:
            self._start_listening()
        elif self.state in (AppState.LISTENING, AppState.DICTATING):
            self._stop_listening()

    def _force_idle(self):
        if self.state == AppState.RECORDING:
            self.recorder.stop()
        elif self.state in (AppState.LISTENING, AppState.DICTATING):
            if self.config.input_mode == "hotkey" and self.state == AppState.DICTATING:
                self._stop_hotkey_stream()
                return
            self._stop_listening()
        self._set_state(AppState.IDLE)

    # -- Streaming mode --

    def _start_listening(self):
        self._save_active_window()
        self._wake_detector = WakeWordDetector(self.config.wake_word)
        self._vad = SimpleVAD(self.config, on_speech_end=self._on_speech_end,
                              on_speech_start=self._on_speech_start)
        self._audio_stream = AudioStream(self.config)
        self._segment_queue = queue.Queue()
        self._segment_worker = threading.Thread(
            target=self._segment_worker_loop, daemon=True,
        )
        self._segment_worker.start()
        try:
            self._audio_stream.start(on_data=self._on_audio_data)
            self._set_state(AppState.LISTENING)
            log.info("Streaming: LISTENING (waiting for wake word '%s')", self.config.wake_word)
        except Exception as e:
            log.error("Failed to start audio stream: %s", e)
            if self._tray:
                self._tray.notify("Error", f"Stream failed: {e}")
            if self._segment_queue:
                self._segment_queue.put(None)
            self._segment_worker = None
            self._segment_queue = None

    def _stop_listening(self):
        self._cancel_silence_timer()
        self._flush_accumulated_text()
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream = None
        self._vad = None
        self._wake_detector = None
        if self._segment_queue:
            self._segment_queue.put(None)
        if self._segment_worker:
            self._segment_worker.join(timeout=5)
            self._segment_worker = None
        self._segment_queue = None
        self._set_state(AppState.IDLE)
        log.info("Streaming: stopped")

    def _start_hotkey_stream(self):
        self._save_active_window()
        self._wake_detector = None
        self._vad = SimpleVAD(self.config, on_speech_end=self._on_speech_end,
                              on_speech_start=self._on_speech_start)
        self._audio_stream = AudioStream(self.config)
        self._accumulated_texts.clear()
        self._segment_queue = queue.Queue()
        self._segment_worker = threading.Thread(
            target=self._segment_worker_loop, daemon=True,
        )
        self._segment_worker.start()
        try:
            self._audio_stream.start(on_data=self._on_audio_data)
            self._set_state(AppState.DICTATING)
            self._play_start_signal()
            log.info("Hotkey+Stream: DICTATING started")
        except Exception as e:
            log.error("Failed to start audio stream: %s", e)
            if self._tray:
                self._tray.notify("Error", f"Stream failed: {e}")
            if self._segment_queue:
                self._segment_queue.put(None)
            self._segment_worker = None
            self._segment_queue = None

    def _stop_hotkey_stream(self):
        self._cancel_silence_timer()
        self._flush_accumulated_text()
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream = None
        self._vad = None
        if self._segment_queue:
            self._segment_queue.put(None)
        if self._segment_worker:
            self._segment_worker.join(timeout=5)
            self._segment_worker = None
        self._segment_queue = None
        self._set_state(AppState.IDLE)
        self._play_end_signal()
        log.info("Hotkey+Stream: stopped")

    def _flush_accumulated_text(self):
        if self._accumulated_texts:
            all_text = " ".join(self._accumulated_texts)
            self._restore_active_window()
            self._inject_text(all_text)
            self._accumulated_texts.clear()

    def _on_audio_data(self, chunk: bytes):
        if self._mute_until and time.time() < self._mute_until:
            return
        if self._vad:
            self._vad.feed(chunk)

    def _on_speech_start(self):
        if self.state == AppState.DICTATING:
            self._cancel_silence_timer()
            log.debug("Silence timer cancelled (speech started)")

    def _on_speech_end(self, pcm_data: bytes):
        if self.state == AppState.DICTATING:
            self._cancel_silence_timer()
        if self._segment_queue:
            self._segment_queue.put(pcm_data)

    def _segment_worker_loop(self):
        while True:
            try:
                pcm_data = self._segment_queue.get(timeout=1.0)
            except queue.Empty:
                if not self._segment_queue:
                    break
                continue
            if pcm_data is None:
                break
            self._process_speech_segment(pcm_data)

    def _process_speech_segment(self, pcm_data: bytes):
        duration_s = len(pcm_data) / (16000 * 2)
        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="whisper-stream-")
        os.close(fd)
        try:
            _write_wav(wav_path, pcm_data)
            model = None
            if self.state == AppState.LISTENING and self.config.wake_model:
                model = self.config.wake_model
            text = self.transcriber.transcribe(wav_path, model=model,
                                               duration_s=duration_s)
            if not text:
                return

            log.info("Stream segment: %r (state=%s)", text[:80], self.state.value)

            if self._wake_detector and self._wake_detector.contains_wake_word(text):
                remaining = self._wake_detector.strip_wake_word(text)

                if self.state == AppState.LISTENING:
                    self._cancel_silence_timer()
                    self._accumulated_texts.clear()
                    self.state = AppState.DICTATING
                    self._marshal_set_state(AppState.DICTATING)
                    self._play_start_signal()
                    log.info("Streaming: DICTATING (wake word detected)")
                    self._marshal_notify("whisper.linux", "Dictation started")
                elif self.state == AppState.DICTATING:
                    self._cancel_silence_timer()
                    if remaining:
                        if self.config.output_mode == "stream":
                            self._restore_active_window()
                            self._inject_text(remaining)
                        else:
                            self._accumulated_texts.append(remaining)
                    self._flush_accumulated_text()
                    self.state = AppState.LISTENING
                    self._marshal_set_state(AppState.LISTENING)
                    log.info("Streaming: LISTENING (wake word \u2192 stop dictation)")
                    self._marshal_notify("whisper.linux", "Dictation paused")
                return

            if self.state == AppState.DICTATING:
                if self.config.output_mode == "stream":
                    self._restore_active_window()
                    self._inject_text(text)
                else:
                    self._accumulated_texts.append(text)
                if self.config.input_mode == "listen":
                    self._reset_silence_timer()
                    log.debug("Silence timer started after segment (%.1fs)",
                              self.config.silence_timeout)
                if self.config.notification:
                    preview = text[:80] + ("..." if len(text) > 80 else "")
                    self._marshal_notify("whisper.linux", preview)

        except Exception as e:
            log.error("Stream segment processing failed: %s", e)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    def _marshal_set_state(self, state: AppState):
        if self._qt_app:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._set_state(state))
        else:
            self._set_state(state)

    def _marshal_notify(self, title: str, message: str):
        if not self._tray:
            return
        if self._qt_app:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._tray.notify(title, message))
        else:
            self._tray.notify(title, message)

    def _reset_silence_timer(self):
        self._cancel_silence_timer()
        self._silence_timer = threading.Timer(
            self.config.silence_timeout,
            self._on_silence_timeout,
        )
        self._silence_timer.daemon = True
        self._silence_timer.start()

    def _cancel_silence_timer(self):
        if self._silence_timer:
            self._silence_timer.cancel()
            self._silence_timer = None

    def _on_silence_timeout(self):
        if self.state == AppState.DICTATING:
            if self._vad and self._vad._in_speech:
                log.debug("Silence timeout skipped (speech active), restarting")
                self._reset_silence_timer()
                return
            log.info("Streaming: silence timeout \u2192 LISTENING")
            self._flush_accumulated_text()
            self.state = AppState.LISTENING
            self._marshal_set_state(AppState.LISTENING)
            self._marshal_notify("whisper.linux", "Dictation paused (silence)")
            self._play_end_signal()

    def _play_sound(self, sounds: tuple):
        """Play the first available sound file."""
        players = ("pw-play", "paplay", "canberra-gtk-play")
        for sound in sounds:
            if not os.path.isfile(sound):
                continue
            for player in players:
                try:
                    subprocess.Popen(
                        [player, sound],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    log.debug("Sound played via %s: %s", player, sound)
                    return
                except FileNotFoundError:
                    continue
        log.debug("No sound player available")

    def _play_start_signal(self):
        if not self.config.end_signal:
            return
        self._mute_until = time.time() + 0.6
        if self._vad:
            self._vad.reset()
        self._play_sound((
            "/usr/share/sounds/freedesktop/stereo/message-new-instant.oga",
            "/usr/share/sounds/freedesktop/stereo/message.oga",
            "/usr/share/sounds/freedesktop/stereo/bell.oga",
        ))

    def _play_end_signal(self):
        if not self.config.end_signal:
            return
        self._mute_until = time.time() + 0.6
        if self._vad:
            self._vad.reset()
        self._play_sound((
            "/usr/share/sounds/freedesktop/stereo/complete.oga",
            "/usr/share/sounds/freedesktop/stereo/bell.oga",
            "/usr/share/sounds/freedesktop/stereo/message.oga",
        ))

    def _set_state(self, state: AppState):
        self.state = state
        if self._tray:
            self._tray.set_state(state)

    def _save_active_window(self):
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                self._prev_window_id = result.stdout.strip()
                log.debug("Saved active window: %s", self._prev_window_id)
        except Exception:
            self._prev_window_id = None

    def _restore_active_window(self):
        if not self._prev_window_id:
            return
        try:
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", self._prev_window_id],
                check=True, timeout=3,
            )
            time.sleep(0.15)
            log.debug("Restored focus to window: %s", self._prev_window_id)
        except Exception as e:
            log.debug("Could not restore window focus: %s", e)

    def _start_recording(self):
        self._save_active_window()
        try:
            self.recorder.start()
            self._set_state(AppState.RECORDING)
        except Exception as e:
            log.error("Failed to start recording: %s", e)
            if self._tray:
                self._tray.notify("Error", f"Recording failed: {e}")

    def _stop_recording(self):
        wav_path = self.recorder.stop()
        if not wav_path:
            self._set_state(AppState.IDLE)
            return
        self._set_state(AppState.PROCESSING)
        t = threading.Thread(target=self._transcribe_and_inject, args=(wav_path,), daemon=True)
        t.start()

    def _transcribe_and_inject(self, wav_path: str):
        try:
            text = self.transcriber.transcribe(wav_path)
            if text:
                self._restore_active_window()
                self._inject_text(text)
                if self._tray and self.config.notification:
                    preview = text[:80] + ("..." if len(text) > 80 else "")
                    self._tray.notify("whisper.linux", preview)
            else:
                log.info("Empty transcription")
                if self._tray:
                    self._tray.notify("whisper.linux", "(no speech detected)")
        except Exception as e:
            log.error("Transcription/injection failed: %s", e)
            if self._tray:
                self._tray.notify("Error", str(e)[:100])
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
            self._set_state(AppState.IDLE)
            self._play_end_signal()

    def quit(self):
        log.info("Shutting down")
        self._cancel_silence_timer()
        if self._audio_stream:
            self._audio_stream.stop()
        if self._segment_queue:
            self._segment_queue.put(None)
        if self._segment_worker:
            self._segment_worker.join(timeout=3)
        self.recorder.cleanup()
        self.remove_pid()
        if self._qt_app:
            self._qt_app.quit()

    # -- Main entry --

    def run(self):
        from PyQt5.QtWidgets import QApplication

        self._qt_app = QApplication(sys.argv)
        self._qt_app.setQuitOnLastWindowClosed(False)

        self.write_pid()
        self._setup_signals()

        from PyQt5.QtCore import QTimer
        signal_timer = QTimer()
        signal_timer.timeout.connect(lambda: None)
        signal_timer.start(200)

        self._tray = TrayIcon(self)
        self._set_state(AppState.IDLE)

        log.info("whisper.linux started (pid %d)", os.getpid())
        log.info("  whisper-cli : %s", self.config.whisper_cli)
        log.info("  model       : %s", self.config.model)
        log.info("  models_dir  : %s", self.config.models_dir)
        log.info("  language    : %s", self.config.language)
        log.info("  threads     : %d", self.config.threads)
        log.info("  gpu_device  : %d", self.config.gpu_device)
        log.info("  audio_device: %s", self.config.audio_device)
        log.info("  display     : %s", self.config.display_server)
        log.info("  paste_keys  : %s", self.config.paste_keys)
        log.info("  clipboard   : %s", self.config.use_clipboard_fallback)
        log.info("  input_mode  : %s", self.config.input_mode)
        log.info("  output_mode : %s", self.config.output_mode)
        log.info("  wake_word   : %s", self.config.wake_word)
        log.info("  wake_model  : %s", self.config.wake_model or "(same as main)")
        log.info("  voice_cmds  : %s", self.config.voice_commands)

        if self.config.input_mode == "listen":
            from PyQt5.QtCore import QTimer as _QT
            _QT.singleShot(0, self.toggle)

        try:
            sys.exit(self._qt_app.exec_())
        finally:
            self.remove_pid()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def send_toggle():
    pid = WhisperLinuxApp.read_pid()
    if pid is None:
        print("whisper-linux is not running.", file=sys.stderr)
        sys.exit(1)
    os.kill(pid, signal.SIGUSR1)
    log.info("Sent SIGUSR1 to pid %d", pid)


def main():
    parser = argparse.ArgumentParser(
        description="whisper.linux \u2014 Voice typing for Linux desktop",
    )
    parser.add_argument("--toggle", action="store_true",
                        help="Toggle recording on a running instance (sends SIGUSR1)")
    parser.add_argument("--language", "-l", default=None,
                        help="Override transcription language (e.g. ru, en, auto)")
    parser.add_argument("--model", "-m", default=None,
                        help="Override model path")
    parser.add_argument("--input-mode", choices=["hotkey", "listen"], default=None,
                        help="Override input mode (hotkey or listen)")
    parser.add_argument("--output-mode", choices=["batch", "stream"], default=None,
                        help="Override output mode (batch or stream)")
    parser.add_argument("--stream", action="store_true",
                        help="Shortcut for --output-mode stream")
    parser.add_argument("--wake-word", default=None,
                        help="Override wake word for listen mode")
    parser.add_argument("--wake-model", default=None,
                        help="Override wake model (lighter model for wake word detection)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.toggle:
        send_toggle()
        return

    existing_pid = WhisperLinuxApp.read_pid()
    if existing_pid is not None:
        print(f"whisper-linux is already running (pid {existing_pid}).", file=sys.stderr)
        print("Use --toggle to start/stop recording, or kill the existing instance.", file=sys.stderr)
        sys.exit(1)

    config = Config()
    if args.language:
        config.language = args.language
    if args.model:
        config.model = args.model
    if args.stream:
        config.output_mode = "stream"
    if args.input_mode:
        config.input_mode = args.input_mode
    if args.output_mode:
        config.output_mode = args.output_mode
    if args.wake_word:
        config.wake_word = args.wake_word
    if args.wake_model:
        config.wake_model = args.wake_model

    if not config.model or not os.path.isfile(config.model):
        if config.model:
            log.warning("Model not found: %s — searching for alternatives", config.model)
        from .config import _find_model
        fallback = _find_model(config.model_search_dirs)
        if fallback:
            log.info("Using model: %s", fallback)
            config.model = fallback
            config.save()
        else:
            print("Error: No model found. Run install.sh or set model path in config.", file=sys.stderr)
            sys.exit(1)

    if config.wake_model and not os.path.isfile(config.wake_model):
        log.warning("Wake model not found: %s — using main model", config.wake_model)
        config.wake_model = ""

    app = WhisperLinuxApp(config)
    app.run()


if __name__ == "__main__":
    main()
